import argparse
import os
import traceback
from pathlib import Path
import scipy as sp
import numpy as np

from utils.logger import create_logger
from utils.mail import send_email_notification
from utils.metrics import calculate_average_metric
from utils.utils import str2bool


class SignificanceConfig(object):
    def __init__(self, dataset, model_1, model_2, pid_1, pid_2, server_run):
        self.dataset = dataset
        self.model_1 = model_1
        self.model_2 = model_2
        self.pid_1 = pid_1
        self.pid_2 = pid_2
        self.server_run = server_run

def RunSignificanceTest(file_name, config):
    train_data = None
    if config.dataset == 0:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, 'synthetic', config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, 'synthetic', config.pid_2))
    if config.dataset == 1:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, Path(file_name).stem, config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, Path(file_name).stem, config.pid_2))
    if config.dataset == 2:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, Path(file_name).stem,
                                                           config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, Path(file_name).stem,
                                                           config.pid_2))
    if config.dataset == 31 or config.dataset == 32 or config.dataset == 33 or config.dataset == 34 or config.dataset\
            == 35:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, Path(file_name).stem,
                                                           config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, Path(file_name).stem,
                                                           config.pid_2))
    if config.dataset == 41 or config.dataset == 42 or config.dataset == 43 or config.dataset == 44 or config.dataset\
            == 45 or config.dataset == 46:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, Path(file_name).stem,
                                                           config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, Path(file_name).stem,
                                                           config.pid_2))
    if config.dataset == 51 or config.dataset == 52 or config.dataset == 53 or config.dataset == 54 or config.dataset\
            == 55 or config.dataset == 56 or config.dataset == 57:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, Path(file_name).stem,
                                                           config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, Path(file_name).stem,
                                                           config.pid_2))
    if config.dataset == 61 or config.dataset == 62 or config.dataset == 63 or config.dataset == 64 or config.dataset\
            == 65 or config.dataset == 66 or config.dataset == 67:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, Path(file_name).stem,
                                                           config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, Path(file_name).stem,
                                                           config.pid_2))
    if config.dataset == 71 or config.dataset == 72 or config.dataset == 73:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, Path(file_name).stem,
                                                           config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, Path(file_name).stem,
                                                           config.pid_2))
    if config.dataset == 81 or config.dataset == 82 or config.dataset == 83 or config.dataset == 84 or config.dataset\
            == 85 or config.dataset == 86 or config.dataset == 87 or config.dataset == 88 or config.dataset == 89 or \
            config.dataset == 90:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, Path(file_name).stem,
                                                           config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, Path(file_name).stem,
                                                           config.pid_2))
    if config.dataset == 91 or config.dataset == 92 or config.dataset == 93 or config.dataset == 94 or config.dataset\
            == 95 or config.dataset == 96 or config.dataset == 97:
        dec_1 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_1, Path(file_name).stem,
                                                           config.pid_1))
        dec_2 = np.load(
            './outputs/NPY/{}/Dec_{}_{}_pid={}.npy'.format(config.dataset, config.model_2, Path(file_name).stem,
                                                           config.pid_2))

    dec_1 = np.random.choice(np.mean(dec_1, axis=1), 20)
    dec_2 = np.random.choice(np.mean(dec_2, axis=1), 20)
    stat, pvalue = sp.stats.ttest_ind(dec_1, dec_2)
    return stat, pvalue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--model_1', type=str, default='VQRAE')
    parser.add_argument('--model_2', type=str, default='VQRAE')
    parser.add_argument('--pid_1', type=int, default=1)
    parser.add_argument('--pid_2', type=int, default=1)
    parser.add_argument('--server_run', type=str2bool, default=False)
    args = parser.parse_args()

    config = SignificanceConfig(dataset=args.dataset, model_1=args.model_1, model_2=args.model_2, pid_1=args.pid_1,
                                pid_2=args.pid_2, server_run=args.server_run)

    train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                           train_logger_name='significance_test_train_logger',
                                                           file_logger_name='significance_test_file_logger',
                                                           meta_logger_name='significance_test_meta_logger',
                                                           model_name=args.model_1 + '_' + args.model_2,
                                                           pid=str(args.pid_1) + '_' + str(args.pid_2))

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
        if args.server_run:
            try:
                stats, pvalue = RunSignificanceTest(file_name=file_name, config=config)
                meta_logger.info('dataset = {}'.format(args.dataset))
                meta_logger.info('file_name = {}'.format(file_name))
                meta_logger.info('model_1 = {}'.format(args.model_1))
                meta_logger.info('model_2 = {}'.format(args.model_2))
                meta_logger.info('pid_1 = {}'.format(args.pid_1))
                meta_logger.info('pid_2 = {}'.format(args.pid_2))
                meta_logger.info('statistics = {}'.format(stats))
                meta_logger.info('pvalue = {}'.format(pvalue))
            except Exception as e:
                send_email_notification(
                    subject='significance_test application CRASHED! dataset error: {}, file error: {}, '
                            'model_1: {}, model_2: {}pid_1: {}, pid_2: {}'.format(
                        args.dataset, file_name, args.model_1, args.model_2, args.pid_1, args.pid_2),
                    message=str(traceback.format_exc()) + str(e))
        else:
            stats, pvalue = RunSignificanceTest(file_name=file_name, config=config)
            meta_logger.info('dataset = {}'.format(args.dataset))
            meta_logger.info('file_name = {}'.format(file_name))
            meta_logger.info('model_1 = {}'.format(args.model_1))
            meta_logger.info('model_2 = {}'.format(args.model_2))
            meta_logger.info('pid_1 = {}'.format(args.pid_1))
            meta_logger.info('pid_2 = {}'.format(args.pid_2))
            meta_logger.info('statistics = {}'.format(stats))
            meta_logger.info('pvalue = {}'.format(pvalue))

    if args.dataset == 1:
        file_name = './outputs/NPY/GD/data/Genesis_AnomalyLabels.csv'
        if args.server_run:
            try:
                stats, pvalue = RunSignificanceTest(file_name=file_name, config=config)
                meta_logger.info('dataset = {}'.format(args.dataset))
                meta_logger.info('file_name = {}'.format(file_name))
                meta_logger.info('model_1 = {}'.format(args.model_1))
                meta_logger.info('model_2 = {}'.format(args.model_2))
                meta_logger.info('pid_1 = {}'.format(args.pid_1))
                meta_logger.info('pid_2 = {}'.format(args.pid_2))
                meta_logger.info('statistics = {}'.format(stats))
                meta_logger.info('pvalue = {}'.format(pvalue))
            except Exception as e:
                send_email_notification(
                    subject='significance_test application CRASHED! dataset error: {}, file error: {}, '
                            'model_1: {}, model_2: {}pid_1: {}, pid_2: {}'.format(
                        args.dataset, file_name, args.model_1, args.model_2, args.pid_1, args.pid_2),
                    message=str(traceback.format_exc()) + str(e))
        else:
            stats, pvalue = RunSignificanceTest(file_name=file_name, config=config)
            meta_logger.info('dataset = {}'.format(args.dataset))
            meta_logger.info('file_name = {}'.format(file_name))
            meta_logger.info('model_1 = {}'.format(args.model_1))
            meta_logger.info('model_2 = {}'.format(args.model_2))
            meta_logger.info('pid_1 = {}'.format(args.pid_1))
            meta_logger.info('pid_2 = {}'.format(args.pid_2))
            meta_logger.info('statistics = {}'.format(stats))
            meta_logger.info('pvalue = {}'.format(pvalue))

    if args.dataset == 2:
        file_name = './data/HSS/data/HRSS_anomalous_standard.csv'
        if args.server_run:
            try:
                stats, pvalue = RunSignificanceTest(file_name=file_name, config=config)
                meta_logger.info('dataset = {}'.format(args.dataset))
                meta_logger.info('file_name = {}'.format(file_name))
                meta_logger.info('model_1 = {}'.format(args.model_1))
                meta_logger.info('model_2 = {}'.format(args.model_2))
                meta_logger.info('pid_1 = {}'.format(args.pid_1))
                meta_logger.info('pid_2 = {}'.format(args.pid_2))
                meta_logger.info('statistics = {}'.format(stats))
                meta_logger.info('pvalue = {}'.format(pvalue))
            except Exception as e:
                send_email_notification(
                    subject='significance_test application CRASHED! dataset error: {}, file error: {}, '
                            'model_1: {}, model_2: {}pid_1: {}, pid_2: {}'.format(
                        args.dataset, file_name, args.model_1, args.model_2, args.pid_1, args.pid_2),
                    message=str(traceback.format_exc()) + str(e))
        else:
            stats, pvalue = RunSignificanceTest(file_name=file_name, config=config)
            meta_logger.info('dataset = {}'.format(args.dataset))
            meta_logger.info('file_name = {}'.format(file_name))
            meta_logger.info('model_1 = {}'.format(args.model_1))
            meta_logger.info('model_2 = {}'.format(args.model_2))
            meta_logger.info('pid_1 = {}'.format(args.pid_1))
            meta_logger.info('pid_2 = {}'.format(args.pid_2))
            meta_logger.info('statistics = {}'.format(stats))
            meta_logger.info('pvalue = {}'.format(pvalue))

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
                s_statistics = []
                s_pvalue = []
                for file in files:
                    file_name = os.path.join(root, file)
                    if args.server_run:
                        try:
                            stats, pvalue = RunSignificanceTest(file_name=file_name, config=config)
                            file_logger.info('dataset = {}'.format(args.dataset))
                            file_logger.info('file_name = {}'.format(file_name))
                            file_logger.info('model_1 = {}'.format(args.model_1))
                            file_logger.info('model_2 = {}'.format(args.model_2))
                            file_logger.info('pid_1 = {}'.format(args.pid_1))
                            file_logger.info('pid_2 = {}'.format(args.pid_2))
                            file_logger.info('statistics = {}'.format(stats))
                            file_logger.info('pvalue = {}'.format(pvalue))
                            s_statistics.append(stats)
                            s_pvalue.append(pvalue)
                        except Exception as e:
                            send_email_notification(
                                subject='significance_test application CRASHED! dataset error: {}, file error: {}, '
                                        'model_1: {}, model_2: {}pid_1: {}, pid_2: {}'.format(
                                    args.dataset, file_name, args.model_1, args.model_2, args.pid_1, args.pid_2),
                                message=str(traceback.format_exc()) + str(e))
                            continue
                    else:
                        stats, pvalue = RunSignificanceTest(file_name=file_name, config=config)
                        s_statistics.append(stats)
                        s_pvalue.append(pvalue)
                        file_logger.info('dataset = {}'.format(args.dataset))
                        file_logger.info('file_name = {}'.format(file_name))
                        file_logger.info('model_1 = {}'.format(args.model_1))
                        file_logger.info('model_2 = {}'.format(args.model_2))
                        file_logger.info('pid_1 = {}'.format(args.pid_1))
                        file_logger.info('pid_2 = {}'.format(args.pid_2))
                        file_logger.info('statistics = {}'.format(stats))
                        file_logger.info('pvalue = {}'.format(pvalue))
                        s_statistics.append(stats)
                        s_pvalue.append(pvalue)
                avg_stats = calculate_average_metric(s_statistics)
                meta_logger.info('avg_stats = {}'.format(avg_stats))
                avg_pvalue = calculate_average_metric(s_pvalue)
                meta_logger.info('avg_pvalue = {}'.format(avg_pvalue))


    if args.server_run:
        send_email_notification(
            subject='visualization application FINISHED! dataset : {}, file: {}, model_1: {}, model_2: {}, pid_1: {}, ' \
                    'pid_2: {}'.format(args.dataset, file_name, args.model_1, args.model_2, args.pid_1, args.pid_2),
            message='visualization application FINISHED! dataset: {}, file: {}, model_1: {}, model_2: {}, pid_1: {}, '
                    'pid_2: {}'.format(args.dataset, file_name, args.model_1, args.model_2, args.pid_1, args.pid_2))
