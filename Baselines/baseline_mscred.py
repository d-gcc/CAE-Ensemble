import argparse
import os
import sqlite3
import traceback
from pathlib import Path
from statistics import mean
import matplotlib as mpl
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, fbeta_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from utils.config import MSCREDConfig
from utils.outputs import CAEOutput
from utils.logger import create_logger
from utils.mail import send_email_notification
from utils.device import get_free_device
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as LA
import pandas as pd
from torch.autograd import Variable
from torch.optim import lr_scheduler, Adam
from utils.data_provider import get_loader, generate_synthetic_dataset, read_GD_dataset, read_HSS_dataset, \
    read_S5_dataset, read_NAB_dataset, read_2D_dataset, read_ECG_dataset, read_SMD_dataset, read_SMAP_dataset, \
    read_MSL_dataset, read_WADI_dataset, read_SWAT_dataset
from utils.metrics import calculate_average_metric, zscore, create_label_based_on_zscore, \
    MetricsResult, create_label_based_on_quantile
from utils.utils import str2bool
from utils.metrics_insert import CalculateMetrics
import pymssql
import time
import logging

#Replicability
random_seed = 0

np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size=1, padding=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size=self.kernel_size,
                               padding=self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state, prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

def signature_matrices_generation(raw_data, gap_time, win):
    """
    Generation signature matrices according win_size and gap_time, the size of raw_data is n * T, n is the number of
    time series, T is the length of time series.
    To represent the inter-correlations between different pairs of time series in a multivariate time series segment
    from t − w to t, we construct an n × n signature matrix Mt based upon the pairwise inner-product of two time series
    within this segment.
    :param win: the length of the time series segment
    :return: the signature matrices
    """
    series_number = raw_data.shape[0]
    series_length = raw_data.shape[1]

    signature_matrices_number = int(series_length / gap_time)

    if win == 0:
        print("The size of win cannot be 0")

    raw_data = np.asarray(raw_data)
    signature_matrices = np.zeros((signature_matrices_number, series_number, series_number))

    for t in range(win, signature_matrices_number):
        raw_data_t = raw_data[:, t - win:t]
        signature_matrices[t] = np.dot(raw_data_t, raw_data_t.T) / win

    return signature_matrices

def generate_train_test(signature_matrices, signature_matrices_number, step_max, train_start_id, test_start_id, series_number):
    """
    Generate train and test dataset, and store them to ../data/train/train.npy and ../data/test/test.npy
    :param signature_matrices:
    :return:
    """
    train_dataset = []
    test_dataset = []

    for data_id in range(signature_matrices_number):
        index = data_id - step_max + 1
        if data_id < train_start_id:
            continue
        index_dataset = signature_matrices[:, index:index + step_max]
        if data_id < test_start_id:
            train_dataset.append(index_dataset)
        else:
            test_dataset.append(index_dataset)

    train_dataset = np.asarray(train_dataset)
    train_dataset = np.reshape(train_dataset, [-1, step_max, series_number, series_number,
                                               signature_matrices.shape[0]])
    test_dataset = np.asarray(test_dataset)
    test_dataset = np.reshape(test_dataset, [-1, step_max, series_number, series_number,
                                             signature_matrices.shape[0]])

    return train_dataset, test_dataset


class CNNEncoderLayers(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(CNNEncoderLayers, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim

        self.first_layer_encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.x_dim, out_channels=self.h_dim, kernel_size=1, stride=1),
            nn.SELU()
        )

        self.second_layer_encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.h_dim, out_channels=self.h_dim * 2, kernel_size=1, stride=1),
            nn.SELU()
        )

        self.third_layer_encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.h_dim * 2, out_channels=self.h_dim * 4, kernel_size=1, stride=1),
            nn.SELU()
        )

        self.fourth_layer_encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.h_dim * 4, out_channels=self.h_dim * 8, kernel_size=1, stride=1),
            nn.SELU()
        )

    def forward(self, x):
        cnn1_out = self.first_layer_encoder(x)
        cnn2_out = self.second_layer_encoder(cnn1_out)
        cnn3_out = self.third_layer_encoder(cnn2_out)
        cnn4_out = self.fourth_layer_encoder(cnn3_out)
        return cnn1_out, cnn2_out, cnn3_out, cnn4_out


class CNNDecoderLayers(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(CNNDecoderLayers, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim

        self.first_layer_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.h_dim * 8, out_channels=self.h_dim * 4, kernel_size=1, stride=1),
            nn.SELU()
        )

        self.second_layer_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.h_dim * 8, out_channels=self.h_dim * 2, kernel_size=1, stride=1),
            nn.SELU()
        )

        self.third_layer_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.h_dim * 4, out_channels=self.h_dim, kernel_size=1, stride=1),
            nn.SELU()
        )

        self.fourth_layer_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.h_dim * 2, out_channels=self.x_dim, kernel_size=1, stride=1),
            nn.SELU()
        )

    def forward(self, lstm1_out, lstm2_out, lstm3_out, lstm4_out):
        dec4 = self.first_layer_decoder(lstm4_out)
        dec4_concat = torch.cat([dec4, lstm3_out], dim=1)
        dec3 = self.second_layer_decoder(dec4_concat)
        dec3_concat = torch.cat([dec3, lstm2_out], dim=1)
        dec2 = self.third_layer_decoder(dec3_concat)
        dec2_concat = torch.cat([dec2, lstm1_out], dim=1)
        dec1 = self.fourth_layer_decoder(dec2_concat)
        return dec1


class CNNLSTMATTLayer(nn.Module):
    def __init__(self, input_channel, output_channel, step_max, kernel_size, padding):
        super(CNNLSTMATTLayer, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.step_max = step_max
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv_lstm_cell = ConvLSTMCell(input_size=self.input_channel, hidden_size=self.output_channel,
                                           kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x):
        h = []
        h_t = torch.zeros([x.shape[0], self.output_channel, x.shape[3], x.shape[4]]).to(device)
        for i in range(x.shape[1]):
            h_t, c_t = self.conv_lstm_cell(x[:, i], h_t)
            h.append(h_t)

            # attention based on inner-product between feature representation of last step and other steps
        attention_w = []
        
        for k in range(self.step_max):
            attention_w.append(torch.sum(h[k] * h[-1]) / self.step_max)
        attention_w = torch.reshape(torch.softmax(torch.stack(attention_w), dim=0), [1, self.step_max])

        outputs = torch.reshape(torch.stack(h, dim=1), [self.step_max, -1])
        outputs = torch.matmul(attention_w, outputs)
        outputs = torch.reshape(outputs, [x.shape[0], x.shape[2], x.shape[3], x.shape[4]])

        return outputs, attention_w


class MSCRED(nn.Module):
    def __init__(self, file_name, config):
        super(MSCRED, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # dim info
        self.x_dim = config.x_dim
        self.h_dim = config.h_dim

        # optimization info
        self.epochs = config.epochs
        self.milestone_epochs = config.milestone_epochs
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.gamma = config.gamma
        self.early_stopping = config.early_stopping
        self.display_epoch = config.display_epoch

        self.step_max = config.step_max
        self.cnn_encoder = CNNEncoderLayers(x_dim=self.x_dim, h_dim=self.h_dim)
        self.cnn_decoder = CNNDecoderLayers(x_dim=self.x_dim, h_dim=self.h_dim)
        self.cnn_lstm_attention_1 = CNNLSTMATTLayer(input_channel=self.h_dim,
                                                    output_channel=self.h_dim,
                                                    kernel_size=1,
                                                    padding=0,
                                                    step_max=self.step_max)
        self.cnn_lstm_attention_2 = CNNLSTMATTLayer(input_channel=self.h_dim * 2,
                                                    output_channel=self.h_dim * 2,
                                                    kernel_size=1,
                                                    padding=0,
                                                    step_max=self.step_max)
        self.cnn_lstm_attention_3 = CNNLSTMATTLayer(input_channel=self.h_dim * 4,
                                                    output_channel=self.h_dim * 4,
                                                    kernel_size=1,
                                                    padding=0,
                                                    step_max=self.step_max)
        self.cnn_lstm_attention_4 = CNNLSTMATTLayer(input_channel=self.h_dim * 8,
                                                    output_channel=self.h_dim * 8,
                                                    kernel_size=1,
                                                    padding=0,
                                                    step_max=self.step_max)

        self.continue_training = config.continue_training

        self.robustness = config.robustness

        # pid
        self.pid = config.pid

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_models/{}/'.format(self.dataset)):
                os.makedirs('./save_models/{}/'.format(self.dataset))
            self.save_model_path = \
                './save_models/{}/MSCRED' \
                '_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_models/{}/MSCRED' \
                '_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None


    def forward(self, x):
        conv1_out, conv2_out, conv3_out, conv4_out = self.cnn_encoder(torch.reshape(x, [x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]]))
        conv1_out = torch.reshape(conv1_out, [-1, 5, self.h_dim, x.shape[3], x.shape[4]])
        conv2_out = torch.reshape(conv2_out, [-1, 5, self.h_dim * 2, x.shape[3], x.shape[4]])
        conv3_out = torch.reshape(conv3_out, [-1, 5, self.h_dim * 4, x.shape[3], x.shape[4]])
        conv4_out = torch.reshape(conv4_out, [-1, 5, self.h_dim * 8, x.shape[3], x.shape[4]])

        conv1_lstm_attention_out, atten_weight_1 = self.cnn_lstm_attention_1(conv1_out)
        conv2_lstm_attention_out, atten_weight_2 = self.cnn_lstm_attention_2(conv2_out)
        conv3_lstm_attention_out, atten_weight_3 = self.cnn_lstm_attention_3(conv3_out)
        conv4_lstm_attention_out, atten_weight_4 = self.cnn_lstm_attention_4(conv4_out)

        # cnn decoder
        deconv_out = self.cnn_decoder(conv1_lstm_attention_out,
                                      conv2_lstm_attention_out,
                                      conv3_lstm_attention_out,
                                      conv4_lstm_attention_out)

        return deconv_out

    def fit(self, train_input, train_label, test_input, test_label):
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)
        loss_fn = nn.MSELoss()
        TN, TP, FN, FP, PRECISION, RECALL, FBETA, PR_AUC, ROC_AUC, CKS = ([] for _ in range(10))
        cae_output = CAEOutput(dec_means=None, train_means=None, validation_means=None, best_TN=None, best_FP=None,
                               best_FN=None, best_TP=None,best_precision=None, best_recall=None, best_fbeta=None,
                               best_pr_auc=None,best_roc_auc=None, best_cks=None)
        # get batch data
        train_data = get_loader(input=train_input, label=None, batch_size=self.batch_size, from_numpy=True,
                                drop_last=True, shuffle=True)
        test_data = get_loader(input=test_input, label=None, batch_size=self.batch_size, from_numpy=True,
                               drop_last=True, shuffle=False)
        if self.load_model == True and self.continue_training == False:
            self.load_state_dict(torch.load(self.load_model_path))
        elif self.load_model == True and self.continue_training == True:
            self.load_state_dict(torch.load(self.load_model_path))
            # train model
            self.train()
            epoch_losses = []
            for epoch in range(self.epochs):
                train_losses = []
                opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # All outputs have shape [sequence_length|rolling_size, batch_size, h_dim|z_dim]
                    # opt.zero_grad()
                    batch_deconv_out = self.forward(batch_x.permute(0, 1, 4, 2, 3).to(device))
                    # loss function: reconstruction error of last step matrix
                    batch_loss = loss_fn(batch_deconv_out, batch_x[:, -1].permute(0, 3, 1, 2).to(device))
                    batch_loss.backward()
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))
                if self.early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
        else:
            # train model
            self.train()
            epoch_losses = []
            for epoch in range(self.epochs):
                train_losses = []
                opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # All outputs have shape [sequence_length|rolling_size, batch_size, h_dim|z_dim]
                    # opt.zero_grad()
                    batch_deconv_out = self.forward(batch_x.permute(0, 1, 4, 2, 3).to(device))
                    # loss function: reconstruction error of last step matrix

                    batch_loss = loss_fn(batch_deconv_out, batch_x[:, -1].permute(0, 3, 1, 2).to(device))
                    batch_loss.backward()
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))
                    
                    self.eval()
                    with torch.no_grad():
                        
                        conv_out = []
                        for i, (batch_x, batch_y) in enumerate(test_data):
                            # batch_x = torch.squeeze(batch_x, dim=0).to(device)
                            batch_deconv_out = self.forward(batch_x.permute(0, 1, 4, 2, 3).to(device))
                            conv_out.append(batch_deconv_out)
                        conv_out = torch.cat(conv_out)
                        
                        errors = []
                        np_reconstruction = conv_out.detach().cpu().numpy()
                        
                        for i in range(np_reconstruction.shape[0]):
                            error = np.square(
                                np.subtract(np.transpose(test_input[:, -1, ...], axes=(0, 3, 1, 2))[i], np_reconstruction[i]))
                            errors.append(error)
                        errors = np.asarray(errors)
                        errors = np.sum(np.reshape(errors, newshape=[errors.shape[0], errors.shape[1], -1]), axis=(1, 2))
                        errors = np.repeat(errors, 5)
                        final_zscore = zscore(errors)

                        pos_label = -1
                        
                        score = errors
                        final_zscore = zscore(errors)
                        np_decision = create_label_based_on_quantile(final_zscore, quantile=98)
                        abnormal_segment = test_label[: np_decision.shape[0]]
                        
                        precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                        recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                        fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
                        cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
                        pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment),
                                                            probas_pred=np.nan_to_num(score), pos_label=pos_label)
                        pr_auc = auc(re, pre)
                        fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), y_score=np.nan_to_num(score),
                                                pos_label=pos_label)
                        roc_auc = auc(fpr, tpr)
                        cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)
                        
                        TN.append(cm[0][0])
                        FP.append(cm[0][1])
                        FN.append(cm[1][0])
                        TP.append(cm[1][1])
                        PRECISION.append(precision)
                        RECALL.append(recall)
                        FBETA.append(fbeta)
                        PR_AUC.append(pr_auc)
                        ROC_AUC.append(roc_auc)
                        CKS.append(cks)
                    
                if self.early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
        if self.save_model:
            torch.save(self.state_dict(), self.save_model_path)
        # test model
        self.eval()
        with torch.no_grad():
            conv_out = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                # batch_x = torch.squeeze(batch_x, dim=0).to(device)
                batch_deconv_out = self.forward(batch_x.permute(0, 1, 4, 2, 3).to(device))
                conv_out.append(batch_deconv_out)
            conv_out = torch.cat(conv_out)
            
            cae_output = CAEOutput(dec_means=None, train_means=None, validation_means=None,best_TN=max(TN), best_FP=max(FP),
                                   best_FN=max(FN), best_TP=max(TP),best_precision=max(PRECISION), best_recall=max(RECALL),
                                   best_fbeta=max(FBETA),best_pr_auc=max(PR_AUC), best_roc_auc=max(ROC_AUC), best_cks=max(CKS))
            
            return conv_out, cae_output

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
    if config.dataset == 101:
        train_data, abnormal_data, abnormal_label = read_SWAT_dataset(file_name, sampling=0.2)
    if config.dataset == 102 or config.dataset == 103:
        train_data, abnormal_data, abnormal_label = read_WADI_dataset(file_name, sampling=0.2)
        
    original_x_dim = abnormal_data.shape[1]

    train_signature_matrices = []
    test_signature_matrices = []
    win_size = [5, 10, 20]
    # Generation signature matrices according the win size w
    for w in win_size:
        train_signature_matrices.append(signature_matrices_generation(raw_data=abnormal_data.T, gap_time=5, win=w))
        test_signature_matrices.append(signature_matrices_generation(raw_data=abnormal_data.T, gap_time=5, win=w))

    train_signature_matrices = np.asarray(train_signature_matrices)
    test_signature_matrices = np.asarray(test_signature_matrices)
    series_number = train_signature_matrices.shape[2]

    train_dataset = []
    test_dataset = []
    for data_id in range(train_signature_matrices.shape[1]):
        index = data_id - config.step_max + 1
        if index < 0:
            continue
        index_dataset = train_signature_matrices[:, index:index + config.step_max]  # create overlapping with sliding windows
        train_dataset.append(index_dataset)

    for data_id in range(test_signature_matrices.shape[1]):
        index = data_id - config.step_max + 1
        if index < 0:
            continue
        index_dataset = test_signature_matrices[:, index:index + config.step_max]  # create overlapping with sliding windows
        test_dataset.append(index_dataset)

    train_dataset = np.asarray(train_dataset, dtype=np.float32)
    train_dataset = np.reshape(train_dataset, [-1, config.step_max, series_number, series_number, train_signature_matrices.shape[0]])
    test_dataset = np.asarray(test_dataset, dtype=np.float32)
    test_dataset = np.reshape(test_dataset, [-1, config.step_max, series_number, series_number, test_signature_matrices.shape[0]])

    config.x_dim = len(win_size)

    model = MSCRED(file_name=file_name, config=config).to(device)
    reconstruction,cae_output = model.fit(train_input=train_dataset, train_label=abnormal_label, test_input=test_dataset, test_label=abnormal_label)

    execution_time = time.time() - start_time_file
    logging.info(file_name + str(execution_time))
    # %%
    if config.save_output:
        np_reconstruction = reconstruction.detach().cpu().numpy()
        if not os.path.exists('./save_outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./save_outputs/NPY/{}/'.format(config.dataset))
        np.save('./save_outputs/NPY/{}/Dec_MSCRED_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), np_reconstruction)

    errors = []
    for i in range(np_reconstruction.shape[0]):
        error = np.square(
            np.subtract(np.transpose(test_dataset[:, -1, ...], axes=(0, 3, 1, 2))[i], np_reconstruction[i]))
        errors.append(error)
    errors = np.asarray(errors)
    errors = np.sum(np.reshape(errors, newshape=[errors.shape[0], errors.shape[1], -1]), axis=(1, 2))
    errors = np.repeat(errors, 5)
    final_zscore = zscore(errors)
    #np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
    np_decision = create_label_based_on_quantile(final_zscore, quantile=98)

    # TODO metrics computation.

    # %%
    if config.save_figure:
        if not os.path.exists('./save_figures/{}/'.format(config.dataset)):
            os.makedirs('./save_figures/{}/'.format(config.dataset))
        if original_x_dim == 1:
            t = np.arange(0, abnormal_data.shape[0])
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
            plt.plot(errors, alpha=0.7)
            # plt.show()
            plt.savefig('./save_figures/{}/Dec_MSCRED_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in abnormal_label[: np_decision.shape[0]]]
            markersize = [4 if i == 1 else 25 for i in abnormal_label[: np_decision.shape[0]]]
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
            plt.plot(np.squeeze(abnormal_data[: np_decision.shape[0]]), alpha=0.7)
            plt.scatter(t[: np_decision.shape[0]], abnormal_data[: np_decision.shape[0]], s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./save_figures/{}/VisInp_MSCRED_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in np_decision]
            markersize = [4 if i == 1 else 25 for i in np_decision]
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
            plt.plot(np.squeeze(abnormal_data[: np_decision.shape[0]]), alpha=0.7)
            plt.scatter(t[: np_decision.shape[0]], abnormal_data[: np_decision.shape[0]], s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./save_figures/{}/VisOut_MSCRED_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        #try:
        pos_label = -1

        abnormal_segment = abnormal_label[: np_decision.shape[0]]
        score = errors

        model_name = 'MSCRED'
        std_dev_values = [2,2.5,3]
        quantile_values = [97,98,99]
        k_values = [0.5,1,2,4,5,10]

        #CalculateMetrics(abnormal_segment, score, pos_label, std_dev_values, quantile_values, k_values, model_name, int(config.pid), int(config.dataset), Path(file_name).stem)

        cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
        precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
        recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
        fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
        pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment), 
                                            probas_pred=np.nan_to_num(score), pos_label=pos_label)
        pr_auc = auc(re, pre)
        fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), 
                                y_score=np.nan_to_num(score), pos_label=pos_label)
        roc_auc = auc(fpr, tpr)       
        cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)

        conn = sqlite3.connect('./experimentsMSCRED.db')
        cursor_obj = conn.cursor()
        insert_sql = """INSERT or REPLACE into mscred (model_name, pid, settings, dataset, file_name, TN, FP, FN, 
        TP, precision, recall, fbeta, pr_auc, roc_auc, cks, best_TN, best_FP, best_FN, best_TP, best_precision, 
        best_recall, best_fbeta, best_pr_auc, best_roc_auc, best_cks) VALUES('{}', '{}', '{}', '{}', '{}', '{}','{}', '{}',
        '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}','{}', '{}')""".format('MSCRED', config.pid, 'None', config.dataset, Path(file_name).stem, cm[0][0], cm[0][1], cm[1][0],cm[1][1], precision, recall, fbeta, pr_auc, roc_auc, cks, cae_output.best_TN, cae_output.best_FP,cae_output.best_FN, cae_output.best_TP, cae_output.best_precision, cae_output.best_recall,cae_output.best_fbeta, cae_output.best_pr_auc, cae_output.best_roc_auc, cae_output.best_cks)
        cursor_obj.execute(insert_sql)
        conn.commit()

        metrics_result = MetricsResult(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1], precision=precision, recall=recall, fbeta=fbeta, pr_auc=pr_auc, roc_auc=roc_auc, cks=cks)
        return metrics_result
       # except:
       #     pass

if __name__ == '__main__':
    conn = sqlite3.connect('./experimentsMSCRED.db')
    cursor_obj = conn.cursor()
    cursor_obj.execute('''CREATE TABLE IF NOT EXISTS mscred (model_name, pid, settings, dataset, file_name, TN, FP, FN, 
            TP, precision, recall, fbeta, pr_auc, roc_auc, cks, best_TN, best_FP, best_FN, best_TP, best_precision, 
            best_recall, best_fbeta, best_pr_auc, best_roc_auc, best_cks)''')
    
    conn.commit()
    logging.basicConfig(level=logging.DEBUG, filename="./Log.txt", filemode="a+",format="%(message)s")
    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=3)
    parser.add_argument('--h_dim', type=int, default=16)
    parser.add_argument('--step_max', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--milestone_epochs', type=int, default=50) 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--early_stopping', type=str2bool, default=False)
    parser.add_argument('--display_epoch', type=int, default=10)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=False)
    parser.add_argument('--save_model', type=str2bool, default=False)  # save model
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--continue_training', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--adjusted_points', type=str2bool, default=False)
    parser.add_argument('--use_best_f1', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    if args.load_config:
        config = MSCREDConfig(dataset=None, x_dim=None, h_dim=None, step_max=None, epochs=None, milestone_epochs=None,
                              lr=None, gamma=None, batch_size=None, weight_decay=None, early_stopping=None,
                              display_epoch=None, save_output=None, save_figure=None, save_model=None, load_model=None,
                              continue_training=None, dropout=None, use_spot=None, save_config=None, load_config=None,
                              server_run=None, robustness=None, pid=None, adjusted_points=None, use_best_f1=None)
        try:
            config.import_config('./save_configs/{}/Config_MSCRED_pid={}.json'.format(config.dataset, config.pid))
        except:
            print('There is no config.')
    else:
        config = MSCREDConfig(dataset=args.dataset, x_dim=args.x_dim, h_dim=args.h_dim, step_max=args.step_max,
                              epochs=args.epochs, milestone_epochs=args.milestone_epochs, lr=args.lr, gamma=args.gamma,
                              batch_size=args.batch_size, weight_decay=args.weight_decay, adjusted_points=args.adjusted_points,
                              early_stopping=args.early_stopping, display_epoch=args.display_epoch,
                              save_output=args.save_output, save_figure=args.save_figure, save_model=args.save_model,
                              load_model=args.load_model, continue_training=args.continue_training,
                              dropout=args.dropout, use_spot=args.use_spot, save_config=args.save_config,
                              load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                              pid=args.pid, use_best_f1=args.use_best_f1)
    if args.save_config:
        if not os.path.exists('./save_configs/{}/'.format(config.dataset)):
            os.makedirs('./save_configs/{}/'.format(config.dataset))
        config.export_config('./save_configs/{}/Config_MSCRED_pid={}.json'.format(config.dataset, config.pid))
        # %%
    device = torch.device(get_free_device())

    train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                           train_logger_name='MSCRED_train_logger',
                                                           file_logger_name='MSCRED_file_logger',
                                                           meta_logger_name='MSCRED_meta_logger',
                                                           model_name='mscred',
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
    start_time_file = time.time()
    execution_time = 0
    
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
                    subject='baseline_mscred application CRASHED! dataset error: {}, file error: {}, '
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
                    subject='baseline_mscred application CRASHED! dataset error: {}, file error: {}, '
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
                    subject='baseline_mscred application CRASHED! dataset error: {}, file error: {}, '
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
    if args.dataset == 101:
        path = './data/SWaT/train'
    if args.dataset == 102:
        path = './data/WADI/2017/train'
    if args.dataset == 103:
        path = './data/WADI/2019/train'
        
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
                                subject='baseline_mscred application CRASHED! dataset error: {}, file error: {}, '
                                              'pid: {}'.format(args.dataset, file_name, args.pid),
                                message=str(traceback.format_exc()) + str(e))
                            continue
                    else:
                        start_time_file = time.time()
                        execution_time = 0
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
            subject='baseline_mscred application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid),
            message='baseline_mscred application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid))