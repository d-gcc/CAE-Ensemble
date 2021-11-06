import argparse
import os
import sqlite3
import traceback
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score, roc_curve, auc, \
    precision_recall_curve, cohen_kappa_score
from utils.config import MPConfig
from utils.data_provider import read_GD_dataset, read_HSS_dataset, read_S5_dataset, read_NAB_dataset, \
    read_ECG_dataset, \
    read_2D_dataset, read_UAH_dataset, read_SMD_dataset, read_SMAP_dataset, read_MSL_dataset, generate_synthetic_dataset
from utils.device import get_free_device
from utils.logger import setup_logger, create_logger
from utils.mail import send_email_notification
from utils.metrics import calculate_metrics, calculate_average_metric, zscore, create_label_based_on_zscore, \
    MetricsResult, create_label_based_on_quantile
from utils.utils import str2bool


def sliding_dot_product(q, t):
    n = t.size
    m = q.size

    # Append t with n zeros
    ta = np.append(t, np.zeros(n))

    # Reverse Q
    qr = np.flip(q, 0)

    # Append qra
    qra = np.append(qr, np.zeros(2 * n - m))

    # Compute FFTs
    qraf = np.fft.fft(qra)
    taf = np.fft.fft(ta)

    # Compute the inverse FFT to the element-wise multiplication of qraf and taf
    qt = np.fft.ifft(np.multiply(qraf, taf))
    return qt[m:n]


def sliding_dot_product_stomp(q, t):
    n = t.size
    m = q.size

    # Append t with n zeros
    ta = np.append(t, np.zeros(n))

    # Reverse Q
    qr = np.flip(q, 0)

    # Append qra
    qra = np.append(qr, np.zeros(2 * n - m))

    # Compute FFTs
    qraf = np.fft.fft(qra)
    taf = np.fft.fft(ta)

    # Compute the inverse FFT to the element-wise multiplication of qraf and taf
    qt = np.fft.ifft(np.multiply(qraf, taf))
    return qt[m - 1:n]


def calculate_distance_profile(q, t, qt, a, sum_q, sum_q2, mean_t, sigma_t):
    n = t.size
    m = q.size

    b = np.zeros(n - m)
    dist = np.zeros(n - m)
    for i in range(0, n - m):
        b[i] = -2 * (qt[i].real - sum_q * mean_t[i]) / sigma_t[i]
        dist[i] = a[i] + b[i] + sum_q2
    return np.sqrt(np.abs(dist))


# The code below takes O(m) for each subsequence
# you should replace it for MASS
def compute_mean_std_for_query(Q):
    # Compute Q stats -- O(n)
    sumQ = np.sum(Q)
    sumQ2 = np.sum(np.power(Q, 2))
    return sumQ, sumQ2


def pre_compute_mean_std_for_TS(ta, m):
    na = len(ta)
    sum_t = np.zeros(na - m)
    sum_t2 = np.zeros(na - m)

    # Compute the stats for t
    cumulative_sum_t = np.cumsum(ta)
    cumulative_sum_t2 = np.cumsum(np.power(ta, 2))
    for i in range(na - m):
        sum_t[i] = cumulative_sum_t[i + m] - cumulative_sum_t[i]
        sum_t2[i] = cumulative_sum_t2[i + m] - cumulative_sum_t2[i]
    mean_t = np.divide(sum_t, m)
    mean_t2 = np.divide(sum_t2, m)
    mean_t_p2 = np.power(mean_t, 2)
    sigma_t2 = np.subtract(mean_t2, mean_t_p2)
    sigma_t = np.sqrt(sigma_t2)
    return sum_t, sum_t2, mean_t, mean_t2, mean_t_p2, sigma_t, sigma_t2


def pre_compute_mean_std_for_TS_stomp(ta, m):
    na = len(ta)
    # Compute the stats for t
    cumulative_sum_t = np.cumsum(ta)
    cumulative_sum_t2 = np.cumsum(np.power(ta, 2))
    sum_t = (cumulative_sum_t[m - 1:na] - np.concatenate(([0], cumulative_sum_t[0:na - m])))
    sum_t2 = (cumulative_sum_t2[m - 1:na] - np.concatenate(([0], cumulative_sum_t2[0:na - m])))
    mean_t = np.divide(sum_t, m)
    mean_t2 = np.divide(sum_t2, m)
    mean_t_p2 = np.power(mean_t, 2)
    sigma_t2 = np.subtract(mean_t2, mean_t_p2)
    sigma_t = np.sqrt(sigma_t2)
    return sum_t, sum_t2, mean_t, mean_t2, mean_t_p2, sigma_t, sigma_t2


# MUEEN’S ALGORITHM FOR SIMILARITY SEARCH (MASS)
def mass(Q, T, a, meanT, sigmaT):
    # Z-Normalisation
    if np.std(Q) != 0:
        Q = (Q - np.mean(Q)) / np.std(Q)
    QT = sliding_dot_product(Q, T)
    sumQ, sumQ2 = compute_mean_std_for_query(Q)
    return calculate_distance_profile(Q, T, QT, a, sumQ, sumQ2, meanT, sigmaT)


def element_wise_min(Pab, Iab, D, idx, ignore_trivial, m):
    for i in range(0, len(D)):
        if not ignore_trivial or (
                np.abs(idx - i) > m / 2.0):  # if it's a self-join, ignore trivial matches in [-m/2,m/2]
            if D[i] < Pab[i]:
                Pab[i] = D[i]
                Iab[i] = idx
    return Pab, Iab


def stamp(Ta, Tb, m):
    """
    Compute the Matrix Profile between time-series Ta and Tb.
    If Ta==Tb, the operation is a self-join and trivial matches are ignored.

    :param Ta: time-series, np.array
    :param Tb: time-series, np.array
    :param m: subsequence length
    :return: Matrix Profile, Nearest-Neighbor indexes
    """
    nb = len(Tb)
    na = len(Ta)
    Pab = np.ones(na - m) * np.inf
    Iab = np.zeros(na - m)
    idxes = np.arange(nb - m + 1)

    sumT, sumT2, meanT, meanT_2, meanTP2, sigmaT, sigmaT2 = pre_compute_mean_std_for_TS(Ta, m)

    a = np.zeros(na - m)
    for i in range(0, na - m):
        a[i] = (sumT2[i] - 2 * sumT[i] * meanT[i] + m * meanTP2[i]) / sigmaT2[i]

    ignore_trivial = np.atleast_1d(Ta == Tb).all()
    for idx in idxes:
        D = mass(Tb[idx: idx + m], Ta, a, meanT, sigmaT)
        if (ignore_trivial):
            # ignore trivial minimum and  maximum
            minIdx = int(np.maximum(idx - m / 2.0, 0))
            maxIdx = int(np.minimum(idx + m / 2.0, len(D)))
            D[minIdx:maxIdx:1] = np.inf

        Iab[Pab > D] = i
        Pab = np.minimum(Pab, D)
    return Pab, Iab


def stomp(T, m):
    """
    Compute the Matrix Profile with self join for T
    :param T: time-series, np.array
    :param Tb: time-series, np.array
    :param m: subsequence length
    :return: Matrix Profile, Nearest-Neighbor indexes
    """
    epsilon = 1e-2

    n = len(T)

    seq_l = n - m
    _, _, meanT, _, _, sigmaT, _ = pre_compute_mean_std_for_TS_stomp(T, m)

    Pab = np.full(seq_l + 1, np.inf)
    Iab = np.zeros(n - m + 1)
    ignore_trivial = True
    for idx in range(0, seq_l):
        # There's something with normalization
        Q_std = sigmaT[idx] if sigmaT[idx] > epsilon else epsilon
        if idx == 0:
            QT = sliding_dot_product_stomp(T[0:m], T).real
            QT_first = np.copy(QT)
        else:
            QT[1:] = QT[0:-1] - (T[0:seq_l] * T[idx - 1]) + (T[m:n] * T[idx + m - 1])
            QT[0] = QT_first[idx]

        # Calculate distance profile
        D = (2 * (m - (QT - m * meanT * meanT[idx]) / (Q_std * sigmaT)))
        D[D < epsilon] = 0
        if (ignore_trivial):
            # ignore trivial minimum and  maximum
            minIdx = int(np.maximum(idx - m / 2.0, 0))
            maxIdx = int(np.minimum(idx + m / 2.0, len(D)))
            D[minIdx:maxIdx:1] = np.inf

        Iab[Pab > D] = idx
        np.minimum(Pab, D, Pab)

    np.sqrt(Pab, Pab)
    return Pab, Iab


# Quick Test
# def test_stomp(Ta, m):
#     start_time = time.time()
#
#     Pab, Iab = stomp(Ta, m)
#     print("--- %s seconds ---" % (time.time() - start_time))
#     plot_motif(Ta, Pab, Iab, m)
#     return Pab, Iab


# Quick Test
# def test_stamp(Ta, Tb, m):
#     start_time = time.time()
#
#     Pab, Iab = stamp(Ta, Tb, m)
#     print("--- %s seconds ---" % (time.time() - start_time))
#
#     plot_discord(Ta, Pab, Iab, m, )
#     return Pab, Iab


def plot_motif(Ta, values, indexes, m):
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(Ta, linestyle='--', alpha=0.5)
    plt.xlim((0, len(Ta)))

    print(np.argmax(values))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Top Motif')
    plt.plot(range(np.argmax(values), np.argmax(values) + m), Ta[np.argmax(values):np.argmax(values) + m], c='r',
             label='Top Discord')

    plt.legend(loc='best')
    plt.title('Time-Series')

    plt.subplot(212)
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


def plot_discord(Ta, Tb, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[int(len(Ta) / len(Tb)), 1])

    plt.subplot(gs[0])
    plt.plot(Ta, linestyle='--')
    plt.xlim((0, len(Ta)))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Best Match')
    plt.legend(loc='best')
    plt.title('Time-Series')
    plt.ylim((-3, 3))

    plt.subplot(gs[1])
    plt.plot(Tb)

    plt.title('Query')
    plt.xlim((0, len(Tb)))
    plt.ylim((-3, 3))

    plt.figure()
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.show()


def plot_match(Ta, Tb, values, indexes, m):
    from matplotlib import gridspec
    plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[int(len(Ta) / len(Tb)), 1])

    plt.subplot(gs[0])
    plt.plot(Ta, linestyle='--')
    plt.xlim((0, len(Ta)))

    print(np.argmax(values))

    plt.plot(range(np.argmin(values), np.argmin(values) + m), Ta[np.argmin(values):np.argmin(values) + m], c='g',
             label='Best Match')
    plt.legend(loc='best')
    plt.title('Time-Series')
    plt.ylim((-3, 3))

    plt.subplot(gs[1])
    plt.plot(Tb)

    plt.title('Query')
    plt.xlim((0, len(Tb)))
    plt.ylim((-3, 3))

    plt.figure()
    plt.title('Matrix Profile')
    plt.plot(range(0, len(values)), values, '#ff5722')
    plt.plot(np.argmax(values), np.max(values), marker='x', c='r', ms=10)
    plt.plot(np.argmin(values), np.min(values), marker='^', c='g', ms=10)

    plt.xlim((0, len(Ta)))
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()


def RunModel(file_name, config):
    train_data = None
    if config.dataset == 0:
        abnormal_data, abnormal_label = generate_synthetic_dataset(case=0, N=512, noise=True, verbose=False)
    if config.dataset == 1:
        abnormal_data, abnormal_label = read_GD_dataset(file_name)
    if config.dataset == 2:
        abnormal_data, abnormal_label = read_HSS_dataset(file_name)
    if config.dataset == 31 or config.dataset == 32 or config.dataset == 33 or config.dataset == 34 or config.dataset == 35:
        abnormal_data, abnormal_label = read_S5_dataset(file_name)
    if config.dataset == 41 or config.dataset == 42 or config.dataset == 43 or config.dataset == 44 or config.dataset == 45 or config.dataset == 46:
        abnormal_data, abnormal_label = read_NAB_dataset(file_name)
    if config.dataset == 51 or config.dataset == 52 or config.dataset == 53 or config.dataset == 54 or config.dataset == 55 or config.dataset == 56 or config.dataset == 57:
        train_data, abnormal_data, abnormal_label = read_2D_dataset(file_name)
    if config.dataset == 61 or config.dataset == 62 or config.dataset == 63 or config.dataset == 64 or config.dataset == 65 or config.dataset == 66 or config.dataset == 67:
        abnormal_data, abnormal_label = read_ECG_dataset(file_name)
    if config.dataset == 71 or config.dataset == 72 or config.dataset == 73:
        train_data, abnormal_data, abnormal_label = read_SMD_dataset(file_name)
    if config.dataset == 81 or config.dataset == 82 or config.dataset == 83 or config.dataset == 84 or config.dataset == 85 or config.dataset == 86 or config.dataset == 87 or config.dataset == 88 or config.dataset == 89 or config.dataset == 90:
        train_data, abnormal_data, abnormal_label = read_SMAP_dataset(file_name)
    if config.dataset == 91 or config.dataset == 92 or config.dataset == 93 or config.dataset == 94 or config.dataset == 95 or config.dataset == 96 or config.dataset == 97:
        train_data, abnormal_data, abnormal_label = read_MSL_dataset(file_name)

    original_x_dim = abnormal_data.shape[1]
    config.x_dim = abnormal_data.shape[1]

    Pab = []
    for i in range(abnormal_data.shape[1]):
        ts = abnormal_data[:, i]
        Pab_i, _ = stomp(ts, config.pattern_size)
        Pab.append(np.nan_to_num(Pab_i))
    Pab = np.sum(Pab, axis=0)
    # final_zscore = zscore(Pab)
    # np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
    np_decision = create_label_based_on_quantile(-Pab, quantile=99)

    if config.save_output:
        if not os.path.exists('./save_outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./save_outputs/NPY/{}/'.format(config.dataset))
        np.save('./save_outputs/NPY/{}/MP_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), Pab)

    # TODO metrics computation.

    # %%
    if config.save_figure:
        if not os.path.exists('./save_figures/{}/'.format(config.dataset)):
            os.makedirs('./save_figures/{}/'.format(config.dataset))
        if original_x_dim == 1:
            plt.figure(figsize=(9, 3))
            plt.plot(ts, color='blue', lw=1.5)
            plt.title('Original Data')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig( './save_figures/{}/Ori_MP_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()

            # Plot decoder output
            plt.figure(figsize=(9, 3))
            plt.plot(Pab, color='blue', lw=1.5)
            plt.title('Profile Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./save_figures/{}/Profile_MP_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()

            t = np.arange(0, abnormal_data.shape[0])
            markercolors = ['blue' for i in range(config.pattern_size-1)] + ['blue' if i == 1 else 'red' for i in abnormal_label[config.pattern_size-1: ]]
            markersize = [4 for i in range(config.pattern_size-1)] + [4 if i == 1 else 25 for i in abnormal_label[config.pattern_size-1:]]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(t[0] - 10, t[-1] + 10)
            ax.set_ylim(-0.10, 1.10)
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.plot(np.squeeze(abnormal_data), alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./save_figures/{}/VisInp_MP_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in np_decision] + ['blue' for i in range(config.pattern_size-1)]
            markersize = [4 if i == 1 else 25 for i in np_decision] + [4 for i in range(config.pattern_size-1)]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            plt.yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlim(t[0] - 10, t[-1] + 10)
            ax.set_ylim(-0.10, 1.10)
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.plot(np.squeeze(abnormal_data), alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./save_figures/{}/VisOut_MP_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        try:
            pos_label = -1
            cm = confusion_matrix(y_true=abnormal_label[config.pattern_size-1: ], y_pred=np_decision, labels=[1, -1])
            precision = precision_score(y_true=abnormal_label[config.pattern_size-1: ], y_pred=np_decision, pos_label=pos_label)
            recall = recall_score(y_true=abnormal_label[config.pattern_size-1: ], y_pred=np_decision, pos_label=pos_label)
            fbeta = fbeta_score(y_true=abnormal_label[config.pattern_size-1: ], y_pred=np_decision, pos_label=pos_label, beta=0.5)
            fpr, tpr, _ = roc_curve(y_true=abnormal_label[config.pattern_size-1: ], y_score=-Pab, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            pre, re, _ = precision_recall_curve(y_true=abnormal_label[config.pattern_size-1: ], probas_pred=-Pab, pos_label=pos_label)
            pr_auc = auc(re, pre)
            cks = cohen_kappa_score(y1=abnormal_label[config.pattern_size-1: ], y2=np_decision)
            settings = config.to_string()
            insert_sql = """INSERT or REPLACE into model (model_name, pid, settings, dataset, file_name, TN, FP, FN, 
            TP, precision, recall, fbeta, pr_auc, roc_auc, cks) VALUES('{}', '{}', '{}', '{}', '{}', '{}', 
            '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')""".format(
                'MP', config.pid, settings, config.dataset, Path(file_name).stem, cm[0][0], cm[0][1], cm[1][0],
                cm[1][1], precision, recall, fbeta, pr_auc, roc_auc, cks)
            cursor_obj.execute(insert_sql)
            conn.commit()
            metrics_result = MetricsResult(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1], precision=precision,
                                           recall=recall, fbeta=fbeta, pr_auc=pr_auc, roc_auc=roc_auc, cks=cks)
            return metrics_result
        except:
            pass


if __name__ == '__main__':
    conn = sqlite3.connect('./experiments.db')
    cursor_obj = conn.cursor()

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--pattern_size', type=int, default=10)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=True)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    if args.load_config:
        config = MPConfig(dataset=None, x_dim=None, pattern_size=None, save_output=None, save_figure=None,
                          use_spot=None, use_last_point=None, save_config=None, load_config=None, server_run=None,
                          robustness=None, pid=None)
        try:
            config.import_config('./save_configs/{}/Config_MP_pid={}.json'.format(config.dataset, config.pid))
        except:
            print('There is no config.')
    else:
        config = MPConfig(dataset=args.dataset, x_dim=args.x_dim, pattern_size=args.pattern_size,
                          save_output=args.save_output, save_figure=args.save_figure, use_spot=args.use_spot,
                          use_last_point=True, save_config=args.save_config, load_config=args.load_config,
                          server_run=args.server_run, robustness=args.robustness, pid=args.pid)
    if args.save_config:
        if not os.path.exists('./save_configs/{}/'.format(config.dataset)):
            os.makedirs('./save_configs/{}/'.format(config.dataset))
        config.export_config('./save_configs/{}/Config_MP_pid={}.json'.format(config.dataset, config.pid))
    # %%
    device = torch.device(get_free_device())
    # %%
    train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                           train_logger_name='mp_train_logger',
                                                           file_logger_name='mp_file_logger',
                                                           meta_logger_name='mp_meta_logger',
                                                           model_name='MP',
                                                           pid=args.pid)

    # logging setting
    file_logger.info('============================')
    for key, value in vars(args).items():
        file_logger.info(key + ' = {}'.format(value))
    file_logger.info('============================')

    meta_logger.info('============================')
    for key, value in vars(args).items():
        meta_logger.info(key + ' = {}'.format(value))
    meta_logger.info('============================')

    path = None

    if args.dataset == 0:
        file_name = 'synthetic'
        file_logger.info(file_name)
        if args.server_run:
            try:
                metrics_result = RunModel(file_name=file_name, config=config)
                meta_logger.info(file_name)
                meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
                meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
                meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
                meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
                meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
                meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
                meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
                meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
                meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
                meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
            except Exception as e:
                send_email_notification(
                    subject='baseline_mp application CRASHED! dataset error: {}, file error: {}, '
                            'pid: {}'.format(args.dataset, file_name, args.pid),
                    message=str(traceback.format_exc()) + str(e))
        else:
            metrics_result = RunModel(file_name=file_name, config=config)
            meta_logger.info(file_name)
            meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
            meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
            meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
            meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
            meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
            meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
            meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
            meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
            meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
            meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
            file_logger.info('Finish')
            # logger.shutdown()
            meta_logger.info('Finish')

    if args.dataset == 1:
        file_name = './data/GD/data/Genesis_AnomalyLabels.csv'
        file_logger.info(file_name)
        if args.server_run:
            try:
                metrics_result = RunModel(file_name=file_name, config=config)
                meta_logger.info(file_name)
                meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
                meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
                meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
                meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
                meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
                meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
                meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
                meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
                meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
                meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
            except Exception as e:
                send_email_notification(
                    subject='baseline_mp application CRASHED! dataset error: {}, file error: {}, '
                            'pid: {}'.format(args.dataset, file_name, args.pid),
                    message=str(traceback.format_exc()) + str(e))
        else:
            metrics_result = RunModel(file_name=file_name, config=config)
            meta_logger.info(file_name)
            meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
            meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
            meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
            meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
            meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
            meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
            meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
            meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
            meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
            meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
            file_logger.info('Finish')
            # logger.shutdown()
            meta_logger.info('Finish')

    if args.dataset == 2:
        file_name = './data/HSS/data/HRSS_anomalous_standard.csv'
        file_logger.info(file_name)
        if args.server_run:
            try:
                metrics_result = RunModel(file_name=file_name, config=config)
                meta_logger.info(file_name)
                meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
                meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
                meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
                meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
                meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
                meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
                meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
                meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
                meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
                meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
            except Exception as e:
                send_email_notification(
                    subject='baseline_mp application CRASHED! dataset error: {}, file error: {}, '
                            'pid: {}'.format(args.dataset, file_name, args.pid),
                    message=str(traceback.format_exc()) + str(e))
        else:
            metrics_result = RunModel(file_name=file_name, config=config)
            meta_logger.info(file_name)
            meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
            meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
            meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
            meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
            meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
            meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
            meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
            meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
            meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
            meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
            file_logger.info('Finish')
            # logger.shutdown()
            meta_logger.info('Finish')

    if args.dataset == 31:
        path = './data/YAHOO/data/A1Benchmark'
    if args.dataset == 32:
        path = './data/YAHOO/data/A2Benchmark'
    if args.dataset == 33:
        path = './data/YAHOO/data/A3Benchmark'
    if args.dataset == 34:
        path = './data/YAHOO/data/A4Benchmark'
    if args.dataset == 35:
        path = './data/YAHOO/data/Vis'
    if args.dataset == 41:
        path = './data/NAB/data/artificialWithAnomaly'
    if args.dataset == 42:
        path = './data/NAB/data/realAdExchange'
    if args.dataset == 43:
        path = './data/NAB/data/realAWSCloudwatch'
    if args.dataset == 44:
        path = './data/NAB/data/realKnownCause'
    if args.dataset == 45:
        path = './data/NAB/data/realTraffic'
    if args.dataset == 46:
        path = './data/NAB/data/realTweets'
    if args.dataset == 51:
        path = './data/2D/Comb'
    if args.dataset == 52:
        path = './data/2D/Cross'
    if args.dataset == 53:
        path = './data/2D/Intersection'
    if args.dataset == 54:
        path = './data/2D/Pentagram'
    if args.dataset == 55:
        path = './data/2D/Ring'
    if args.dataset == 56:
        path = './data/2D/Stripe'
    if args.dataset == 57:
        path = './data/2D/Triangle'
    if args.dataset == 61:
        path = './data/ECG/chf01'
    if args.dataset == 62:
        path = './data/ECG/chf13'
    if args.dataset == 63:
        path = './data/ECG/ltstdb43'
    if args.dataset == 64:
        path = './data/ECG/ltstdb240'
    if args.dataset == 65:
        path = './data/ECG/mitdb180'
    if args.dataset == 66:
        path = './data/ECG/stdb308'
    if args.dataset == 67:
        path = './data/ECG/xmitdb108'
    if args.dataset == 71:
        path = './data/SMD/machine1/train'
    if args.dataset == 72:
        path = './data/SMD/machine2/train'
    if args.dataset == 73:
        path = './data/SMD/machine3/train'
    if args.dataset == 81:
        path = './data/SMAP/channel1/train'
    if args.dataset == 82:
        path = './data/SMAP/channel2/train'
    if args.dataset == 83:
        path = './data/SMAP/channel3/train'
    if args.dataset == 84:
        path = './data/SMAP/channel4/train'
    if args.dataset == 85:
        path = './data/SMAP/channel5/train'
    if args.dataset == 86:
        path = './data/SMAP/channel6/train'
    if args.dataset == 87:
        path = './data/SMAP/channel7/train'
    if args.dataset == 88:
        path = './data/SMAP/channel8/train'
    if args.dataset == 89:
        path = './data/SMAP/channel9/train'
    if args.dataset == 90:
        path = './data/SMAP/channel10/train'
    if args.dataset == 91:
        path = './data/MSL/channel1/train'
    if args.dataset == 92:
        path = './data/MSL/channel2/train'
    if args.dataset == 93:
        path = './data/MSL/channel3/train'
    if args.dataset == 94:
        path = './data/MSL/channel4/train'
    if args.dataset == 95:
        path = './data/MSL/channel5/train'
    if args.dataset == 96:
        path = './data/MSL/channel6/train'
    if args.dataset == 97:
        path = './data/MSL/channel7/train'

    if path is not None:
        for root, dirs, files in os.walk(path):
            if len(dirs) == 0:
                s_TN = []
                s_FP = []
                s_FN = []
                s_TP = []
                s_precision = []
                s_recall = []
                s_fbeta = []
                s_roc_auc = []
                s_pr_auc = []
                s_cks = []
                s_best_TN = []
                s_best_FP = []
                s_best_FN = []
                s_best_TP = []
                s_best_precision = []
                s_best_recall = []
                s_best_fbeta = []
                s_best_roc_auc = []
                s_best_pr_auc = []
                s_best_cks = []
                for file in files:
                    file_name = os.path.join(root, file)
                    file_logger.info('============================')
                    file_logger.info(file)

                    if args.server_run:
                        try:
                            metrics_result = RunModel(file_name=file_name, config=config)
                            s_TN.append(metrics_result.TN)
                            file_logger.info('TN = {}'.format(metrics_result.TN))
                            s_FP.append(metrics_result.FP)
                            file_logger.info('FP = {}'.format(metrics_result.FP))
                            s_FN.append(metrics_result.FN)
                            file_logger.info('FN = {}'.format(metrics_result.FN))
                            s_TP.append(metrics_result.TP)
                            file_logger.info('TP = {}'.format(metrics_result.TP))
                            s_precision.append(metrics_result.precision)
                            file_logger.info('precision = {}'.format(metrics_result.precision))
                            s_recall.append(metrics_result.recall)
                            file_logger.info('recall = {}'.format(metrics_result.recall))
                            s_fbeta.append(metrics_result.fbeta)
                            file_logger.info('fbeta = {}'.format(metrics_result.fbeta))
                            s_roc_auc.append(metrics_result.roc_auc)
                            file_logger.info('roc_auc = {}'.format(metrics_result.roc_auc))
                            s_pr_auc.append(metrics_result.pr_auc)
                            file_logger.info('pr_auc = {}'.format(metrics_result.pr_auc))
                            s_cks.append(metrics_result.cks)
                            file_logger.info('cks = {}'.format(metrics_result.cks))
                        except Exception as e:
                            send_email_notification(
                                subject='baseline_mp application CRASHED! dataset error: {}, file error: {}, '
                                        'pid: {}'.format(args.dataset, file_name, args.pid),
                                message=str(traceback.format_exc()) + str(e))
                            continue
                    else:
                        metrics_result = RunModel(file_name=file_name, config=config)
                        s_TN.append(metrics_result.TN)
                        file_logger.info('TN = {}'.format(metrics_result.TN))
                        s_FP.append(metrics_result.FP)
                        file_logger.info('FP = {}'.format(metrics_result.FP))
                        s_FN.append(metrics_result.FN)
                        file_logger.info('FN = {}'.format(metrics_result.FN))
                        s_TP.append(metrics_result.TP)
                        file_logger.info('TP = {}'.format(metrics_result.TP))
                        s_precision.append(metrics_result.precision)
                        file_logger.info('precision = {}'.format(metrics_result.precision))
                        s_recall.append(metrics_result.recall)
                        file_logger.info('recall = {}'.format(metrics_result.recall))
                        s_fbeta.append(metrics_result.fbeta)
                        file_logger.info('fbeta = {}'.format(metrics_result.fbeta))
                        s_roc_auc.append(metrics_result.roc_auc)
                        file_logger.info('roc_auc = {}'.format(metrics_result.roc_auc))
                        s_pr_auc.append(metrics_result.pr_auc)
                        file_logger.info('pr_auc = {}'.format(metrics_result.pr_auc))
                        s_cks.append(metrics_result.cks)
                        file_logger.info('cks = {}'.format(metrics_result.cks))
                meta_logger.info(dir)
                avg_TN = calculate_average_metric(s_TN)
                meta_logger.info('avg_TN = {}'.format(avg_TN))
                avg_FP = calculate_average_metric(s_FP)
                meta_logger.info('avg_FP = {}'.format(avg_FP))
                avg_FN = calculate_average_metric(s_FN)
                meta_logger.info('avg_FN = {}'.format(avg_FN))
                avg_TP = calculate_average_metric(s_TP)
                meta_logger.info('avg_TP = {}'.format(avg_TP))
                avg_precision = calculate_average_metric(s_precision)
                meta_logger.info('avg_precision = {}'.format(avg_precision))
                avg_recall = calculate_average_metric(s_recall)
                meta_logger.info('avg_recall = {}'.format(avg_recall))
                avg_fbeta = calculate_average_metric(s_fbeta)
                meta_logger.info('avg_fbeta = {}'.format(avg_fbeta))
                avg_roc_auc = calculate_average_metric(s_roc_auc)
                meta_logger.info('avg_roc_auc = {}'.format(avg_roc_auc))
                avg_pr_auc = calculate_average_metric(s_pr_auc)
                meta_logger.info('avg_pr_auc = {}'.format(avg_pr_auc))
                avg_cks = calculate_average_metric(s_cks)
                meta_logger.info('avg_cks = {}'.format(avg_cks))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
                # meta_logger.shutdown()

    if args.server_run:
        send_email_notification(
            subject='baseline_mp application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid),
            message='baseline_mp application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid))