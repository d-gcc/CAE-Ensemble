import argparse
import os
import traceback
from pathlib import Path
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.data_provider import generate_synthetic_dataset, read_GD_dataset, read_HSS_dataset, read_S5_dataset, \
    read_NAB_dataset, read_2D_dataset, read_ECG_dataset, read_SMD_dataset, read_SMAP_dataset, read_MSL_dataset, \
    rolling_window_2D, cutting_window_2D
from utils.mail import send_email_notification
from utils.utils import str2bool
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class VisualizationConfig(object):
    def __init__(self, dataset, model, preprocessing, use_overlapping, rolling_size, pid):
        self.dataset = dataset
        self.model = model
        self.preprocessing = preprocessing
        self.use_overlapping = use_overlapping
        self.rolling_size = rolling_size
        self.pid = pid


def RunVisualization(file_name, config):
    train_data = None
    if config.dataset == 0:
        abnormal_data, abnormal_label = generate_synthetic_dataset(case=0, N=512, noise=True, verbose=False)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, 'synthetic', config.pid))
    if config.dataset == 1:
        abnormal_data, abnormal_label = read_GD_dataset(file_name)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, Path(file_name).stem, config.pid))
    if config.dataset == 2:
        abnormal_data, abnormal_label = read_HSS_dataset(file_name)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, Path(file_name).stem, config.pid))
    if config.dataset == 31 or config.dataset == 32 or config.dataset == 33 or config.dataset == 34 or config.dataset == 35:
        abnormal_data, abnormal_label = read_S5_dataset(file_name)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, Path(file_name).stem, config.pid))
    if config.dataset == 41 or config.dataset == 42 or config.dataset == 43 or config.dataset == 44 or config.dataset == 45 or config.dataset == 46:
        abnormal_data, abnormal_label = read_NAB_dataset(file_name)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, Path(file_name).stem, config.pid))
    if config.dataset == 51 or config.dataset == 52 or config.dataset == 53 or config.dataset == 54 or config.dataset == 55 or config.dataset == 56 or config.dataset == 57:
        train_data, abnormal_data, abnormal_label = read_2D_dataset(file_name)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, Path(file_name).stem, config.pid))
    if config.dataset == 61 or config.dataset == 62 or config.dataset == 63 or config.dataset == 64 or config.dataset == 65 or config.dataset == 66 or config.dataset == 67:
        abnormal_data, abnormal_label = read_ECG_dataset(file_name)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, Path(file_name).stem, config.pid))
    if config.dataset == 71 or config.dataset == 72 or config.dataset == 73:
        train_data, abnormal_data, abnormal_label = read_SMD_dataset(file_name)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, Path(file_name).stem, config.pid))
    if config.dataset == 81 or config.dataset == 82 or config.dataset == 83 or config.dataset == 84 or config.dataset == 85 or config.dataset == 86 or config.dataset == 87 or config.dataset == 88 or config.dataset == 89 or config.dataset == 90:
        train_data, abnormal_data, abnormal_label = read_SMAP_dataset(file_name)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, Path(file_name).stem, config.pid))
    if config.dataset == 91 or config.dataset == 92 or config.dataset == 93 or config.dataset == 94 or config.dataset == 95 or config.dataset == 96 or config.dataset == 97:
        train_data, abnormal_data, abnormal_label = read_MSL_dataset(file_name)
        latent_output = np.load('./outputs/NPY/{}/Latent_{}_{}_pid={}.npy'.format(config.dataset, config.model, Path(file_name).stem, config.pid))

    label_unroll = abnormal_label[config.rolling_size - 1:]
    scaler = preprocessing.MinMaxScaler()
    if config.model == 'DONUT':
        latent_output = scaler.fit_transform(latent_output)
    if config.model == 'RNNVAE':
        latent_output = scaler.fit_transform(latent_output[:,-1])
    elif config.model == 'OMNIANOMALY':
        latent_output = scaler.fit_transform(latent_output[:,-1])
    elif config.model == 'VQRAE':
        latent_output = scaler.fit_transform(latent_output[:,-1])
    markercolors = ['blue' if i == 1 else 'red' for i in label_unroll]
    markersize = [4 if i == 1 else 25 for i in label_unroll]

    latent_output_embedded_PCA = PCA(n_components=2).fit_transform(latent_output)
    latent_output_embedded_PCA = scaler.fit_transform(latent_output_embedded_PCA)
    x_1PCA = latent_output_embedded_PCA[:, 0]
    y_1PCA = latent_output_embedded_PCA[:, 1]

    latent_output_embedded_TSNE = TSNE(n_components=2).fit_transform(latent_output)
    latent_output_embedded_TSNE = scaler.fit_transform(latent_output_embedded_TSNE)
    x_1TSNE = latent_output_embedded_TSNE[:, 0]
    y_1TSNE = latent_output_embedded_TSNE[:, 1]

    x_2 = latent_output[:, 0]
    y_2 = latent_output[:, 1]
    z_2 = latent_output[:, 2]

    fig1PCA = plt.figure()
    ax = fig1PCA.add_subplot(111)
    ax.scatter(x_1PCA, y_1PCA, c=markercolors, s=markersize)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        './figures/{}/2DLatentPCA_{}_{}_pid={}.png'.format(config.dataset, config.model, Path(file_name).stem, config.pid),
        dpi=300)
    plt.close()

    fig2PCA = plt.figure()
    ax = fig2PCA.add_subplot(111)
    ax.scatter(x_1TSNE, y_1TSNE, c=markercolors, s=markersize)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        './figures/{}/2DLatentTSNE_{}_{}_pid={}.png'.format(config.dataset, config.model, Path(file_name).stem, config.pid),
        dpi=300)
    plt.close()

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.scatter(x_2, y_2, z_2, c=markercolors, s=markersize)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./figures/{}/3DLatent_{}_{}_pid={}.png'.format(config.dataset, config.model, Path(file_name).stem, config.pid), dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--model', type=str, default='DONUT')
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    parser.add_argument('--rolling_size', type=int, default=32)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    config = VisualizationConfig(dataset=args.dataset, model=args.model, preprocessing=args.preprocessing,
                                 use_overlapping=args.use_overlapping, rolling_size=args.rolling_size, pid=args.pid)

    path = None

    if args.dataset == 0:
        file_name = 'synthetic'
        if args.server_run:
            try:
                RunVisualization(file_name=file_name, config=config)
            except Exception as e:
                send_email_notification(
                    subject='visualization application CRASHED! dataset error: {}, file error: {}, '
                            'pid: {}'.format(args.dataset, file_name, args.pid),
                    message=str(traceback.format_exc()) + str(e))
        else:
            RunVisualization(file_name=file_name, config=config)

    if args.dataset == 1:
        file_name = './data/GD/data/Genesis_AnomalyLabels.csv'
        if args.server_run:
            try:
                RunVisualization(file_name=file_name, config=config)
            except Exception as e:
                send_email_notification(
                    subject='visualization application CRASHED! dataset error: {}, file error: {}, '
                            'pid: {}'.format(args.dataset, file_name, args.pid),
                    message=str(traceback.format_exc()) + str(e))
        else:
            RunVisualization(file_name=file_name, config=config)

    if args.dataset == 2:
        file_name = './data/HSS/data/HRSS_anomalous_standard.csv'
        if args.server_run:
            try:
                RunVisualization(file_name=file_name, config=config)
            except Exception as e:
                send_email_notification(
                    subject='visualization application CRASHED! dataset error: {}, file error: {}, '
                            'pid: {}'.format(args.dataset, file_name, args.pid),
                    message=str(traceback.format_exc()) + str(e))
        else:
            RunVisualization(file_name=file_name, config=config)

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
                for file in files:
                    file_name = os.path.join(root, file)
                    if args.server_run:
                        try:
                            RunVisualization(file_name=file_name, config=config)
                        except Exception as e:
                            send_email_notification(
                                subject='visualization application CRASHED! dataset error: {}, file error: {}, '
                            'pid: {}'.format(args.dataset, file_name, args.pid),
                                message=str(traceback.format_exc()) + str(e))
                            continue
                    else:
                        RunVisualization(file_name=file_name, config=config)

    if args.server_run:
        send_email_notification(
            subject='visualization application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid),
            message='visualization application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid))

