#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import sqlite3
import traceback
from pathlib import Path
from statistics import mean
import matplotlib as mpl
from sklearn import preprocessing
import torch.nn.functional as F

from utils.outputs import CAEOutput
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams["font.size"] = 16
import numpy as np
from pathlib import Path
from statistics import mean
import numpy as np
import random
import math
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, auc, precision_recall_curve, roc_curve,     fbeta_score, confusion_matrix
from torch.optim import Adam, lr_scheduler
from utils.config import CAEConfig
from utils.data_provider import read_GD_dataset, read_HSS_dataset, read_S5_dataset, read_NAB_dataset, read_2D_dataset,     read_ECG_dataset, get_loader, generate_synthetic_dataset, read_SMD_dataset, read_SMAP_dataset, read_MSL_dataset,     rolling_window_2D, cutting_window_2D, unroll_window_3D, read_WADI_dataset, read_SWAT_dataset
from utils.device import get_free_device
from utils.logger import create_logger
from utils.mail import send_email_notification
from utils.metrics import calculate_average_metric, MetricsResult, zscore, create_label_based_on_zscore,     create_label_based_on_quantile
from utils.utils import str2bool
from utils.spot import SPOT
from utils.metrics_insert import CalculateMetrics
from utils.validation_insert import ValidationMetrics
import pymssql
import time

#Replicability
random_seed = 0

np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True


# In[2]:


class Encoder(nn.Module):

    def __init__(self, x_dim, h_dim, sequence_length, dropout):
        super(Encoder, self).__init__()
        h_dim_half = int(h_dim / 2)
        self.sequence_length = sequence_length
        self.h_dim = h_dim
        self.x_dim = x_dim
        
        self.linear = nn.Linear(in_features=x_dim, out_features=h_dim_half)
        self.position = nn.Linear(sequence_length, h_dim_half)
        self.convolutions = nn.ModuleList([nn.Conv1d(in_channels=h_dim * 2, out_channels=h_dim * 4,
                                                     kernel_size=3, padding=1) for _ in range(5)])
        self.to_hidden = nn.Linear(h_dim_half, h_dim * 2)
        self.from_hidden = nn.Linear(h_dim * 2, h_dim_half)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        linear_output = self.linear(input)
        position_series = torch.arange(0, self.sequence_length).repeat(input.size()[0], 1).float().to(device)
        position_embedded = self.position(position_series).unsqueeze(1)
        embedded = self.dropout(linear_output + position_embedded)
        convolution_input = self.to_hidden(embedded).permute(0, 2, 1)

        for i, convolution in enumerate(self.convolutions):
            convolution_hidden = F.glu(convolution(self.dropout(convolution_input)), dim=1)
            convolution_hidden = convolution_hidden + convolution_input
            convolution_input = convolution_hidden

        convolution_hidden = self.from_hidden(convolution_input.permute(0, 2, 1))
        residual_connection = convolution_hidden + embedded
        return convolution_hidden, residual_connection


# In[3]:


class Decoder(nn.Module):

    def __init__(self, x_dim, h_dim, sequence_length, dropout):
        super(Decoder, self).__init__()
        h_dim_half = int(h_dim / 2)
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.sequence_length = sequence_length

        self.linear = nn.Linear(in_features=x_dim, out_features=h_dim_half)
        self.position = nn.Linear(sequence_length, h_dim_half)
        self.convolutions = nn.ModuleList([nn.Conv1d(in_channels=h_dim * 2, out_channels=h_dim * 4,
                                                     kernel_size=3) for _ in range(5)])
        self.to_hidden = nn.Linear(h_dim_half, h_dim * 2)
        self.from_hidden = nn.Linear(h_dim * 2, h_dim_half)
        self.attention_to_hidden = nn.Linear(h_dim_half, h_dim * 2)
        self.attention_from_hidden = nn.Linear(h_dim * 2, h_dim_half)
        self.dropout = nn.Dropout(p=dropout)
        self.output = nn.Linear(h_dim_half, x_dim)

    def calculate_attention(self, embedded, convolution_hidden, encoder_convolution_hidden, encoder_residual_connection):
        residual_connection = self.attention_from_hidden(convolution_hidden.permute(0, 2, 1)) + embedded
        attention = F.softmax(torch.matmul(residual_connection, encoder_convolution_hidden.permute(0, 2, 1)), dim=2)
        attention_encoded = self.attention_to_hidden(torch.matmul(attention, encoder_residual_connection))
        attention_connection = convolution_hidden + attention_encoded.permute(0, 2, 1)
        return attention, attention_connection

    def forward(self, target, hidden_encoder, residual_encoder):
        linear_output = self.linear(target)
        position_series = torch.arange(0, self.sequence_length).repeat(target.size()[0], 1).float().to(device)
        position_embedded = self.position(position_series).unsqueeze(1)
        embedded = self.dropout(linear_output + position_embedded)
        convolution_input = self.to_hidden(embedded).permute(0, 2, 1)

        for i, convolution in enumerate(self.convolutions):
            convolution_input = self.dropout(convolution_input)
            padding = torch.zeros(convolution_input.size()[0], self.h_dim * 2, 2).fill_(1).to(device)
            padded_convolution_input = torch.cat((padding, convolution_input), dim=2)
            convolution_hidden = F.glu(convolution(padded_convolution_input), dim=1)
            attention, convolution_hidden = self.calculate_attention(embedded, convolution_hidden, hidden_encoder, residual_encoder)
            convolution_hidden = convolution_hidden + convolution_input
            convolution_input = convolution_hidden

        convolution_hidden = self.from_hidden(convolution_hidden.permute(0, 2, 1))

        output = self.output(convolution_hidden)

        return output


# In[4]:


class DiversityLoss(nn.Module):
    def __init__(self, ensemble_member, diversity_factor):
        super(DiversityLoss, self).__init__()
        self.ensemble_member = ensemble_member
        self.diversity_factor = diversity_factor
        self.loss_function = nn.MSELoss()

    def forward(self, input, target, batch_number, batch_size):
        size_input = input.size()[0]
        norm_results = torch.zeros(1, 1).to(device)

        for i in range(len(self.ensemble_member)):
            batch_segment = self.ensemble_member[i].train_means[batch_number * batch_size:
                                                                batch_number * batch_size + size_input]
            norm_results += math.sqrt(2) / 2 * torch.norm(input - batch_segment, 2) / size_input

        loss_diversity = self.loss_function(input, target) + self.diversity_factor * norm_results / len(
            self.ensemble_member)

        return loss_diversity


# In[5]:


class CAE(nn.Module):
    def __init__(self, file_name, config):
        super(CAE, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # dim info
        self.x_dim = config.x_dim
        self.h_dim = config.h_dim

        # sequence info
        self.preprocessing = config.preprocessing
        self.use_overlapping = config.use_overlapping
        self.use_last_point = config.use_last_point
        self.rolling_size = config.rolling_size

        # optimization info
        self.epochs = config.epochs
        self.milestone_epochs = config.milestone_epochs
        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.early_stopping = config.early_stopping
        self.loss_function = config.loss_function
        self.display_epoch = config.display_epoch
        self.use_clip_norm = config.use_clip_norm
        self.gradient_clip_norm = config.gradient_clip_norm
        
        self.dropout = config.dropout
        self.continue_training = config.continue_training
        self.robustness = config.robustness
        self.validation = config.validation
        self.pid = config.pid

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_models/{}/'.format(self.dataset)):
                os.makedirs('./save_models/{}/'.format(self.dataset))
            self.save_model_path = './save_models/{}/CAE_ENSEMBLE_{}_pid={}.pt'.format(self.dataset, Path(
                self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = './save_models/{}/CAE_ENSEMBLE_{}_pid={}.pt'.format(self.dataset, Path(
                self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

        self.encoder = Encoder(x_dim=self.x_dim, h_dim=self.h_dim, sequence_length=self.rolling_size,
                               dropout=self.dropout)
        self.decoder = Decoder(x_dim=self.x_dim, h_dim=self.h_dim, sequence_length=self.rolling_size,
                               dropout=self.dropout)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Encoder):
                self.encoder = Encoder(batch_size=self.batch_size)
            elif isinstance(m, Decoder):
                self.decoder = Decoder(batch_size=self.batch_size)

    def forward(self, input):
        hidden_state, residual = self.encoder(input)
        decoded_output = self.decoder(input, hidden_state, residual)
        return decoded_output

    def fit(self, train_input, val_input, test_input, test_label, abnormal_data, abnormal_label, original_x_dim, ensemble_member=[]):
        TN, TP, FN, FP, PRECISION, RECALL, FBETA, PR_AUC, ROC_AUC, CKS = ([] for _ in range(10))

        if len(ensemble_member) == 0:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = DiversityLoss(ensemble_member, config.diversity_factor)

        opt = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)

        train_data = get_loader(input=train_input, label=None, batch_size=self.batch_size, from_numpy=True,
                                drop_last=False, shuffle=False)
        test_data = get_loader(input=test_input, label=test_label, batch_size=self.batch_size, from_numpy=True,
                               drop_last=False, shuffle=False)
        if self.load_model == True and self.continue_training == False:
            self.load_state_dict(torch.load(self.load_model_path))
            
        else:
            # train model
            self.train()
            epoch_losses = []
            for epoch in range(self.epochs):
                train_losses = []

                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    batch_x = batch_x.to(device)
                    batch_x_reconstruct = self.forward(batch_x)

                    if len(ensemble_member) == 0:
                        batch_loss = loss_fn(batch_x_reconstruct, batch_x)
                    else:
                        batch_loss = loss_fn(batch_x_reconstruct, batch_x, i, self.batch_size)

                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))
                    self.eval()
                    with torch.no_grad():
                        cat_xs = []
                        for i, (batch_x, batch_y) in enumerate(test_data):
                            batch_x = batch_x.to(device)
                            batch_x_reconstruct = self.forward(batch_x)
                            cat_xs.append(batch_x_reconstruct)

                        if self.preprocessing:
                            cat_xs = torch.cat(cat_xs, dim=0)
                        else:
                            cat_xs = torch.cat(cat_xs, dim=0)

                        cae_output = CAEOutput(dec_means=cat_xs, train_means=None, validation_means=None, 
                                               best_TN=None, best_FP=None,best_FN=None, best_TP=None,
                                               best_precision=None, best_recall=None, best_fbeta=None, best_pr_auc=None,
                                               best_roc_auc=None, best_cks=None)

                        min_max_scaler = preprocessing.StandardScaler()
                        if self.preprocessing:
                            if self.use_overlapping:
                                if self.use_last_point:
                                    dec_mean_unroll = cae_output.dec_means.detach().cpu().numpy()[:, -1]
                                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                                    x_original_unroll = abnormal_data[self.rolling_size - 1:]
                                    abnormal_segment = abnormal_label[self.rolling_size - 1:]
                                else:
                                    dec_mean_unroll = unroll_window_3D(np.reshape(cae_output.dec_means.detach().cpu().numpy(), (-1, self.rolling_size, original_x_dim)))[::-1]
                                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                                    x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                                    abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
                            else:
                                dec_mean_unroll = np.reshape(cae_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
                                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                                abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
                        else:
                            dec_mean_unroll = cae_output.dec_means.detach().cpu().numpy()
                            dec_mean_unroll = np.transpose(np.squeeze(dec_mean_unroll, axis=0))
                            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                            x_original_unroll = abnormal_data
                            abnormal_segment = abnormal_label

                        error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2
                        final_zscore = zscore(error)
                        np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)

                        pos_label = -1

                        precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                        recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                        fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
                        cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
                        pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment),
                                                            probas_pred=np.nan_to_num(error), pos_label=pos_label)
                        pr_auc = auc(re, pre)
                        fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), y_score=np.nan_to_num(error),
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

                        if config.validation == 3:
                            
                            valid_data = get_loader(input=val_input, label=None, batch_size=self.batch_size, from_numpy=True,
                               drop_last=False, shuffle=False)
                            val_train = []
                            for i, (batch_x, batch_y) in enumerate(valid_data):
                                batch_x = batch_x.to(device)
                                batch_x_reconstruct = self.forward(batch_x)
                                val_train.append(batch_x_reconstruct)

                            val_train = torch.cat(val_train, dim=0)

                            min_max_scaler = preprocessing.StandardScaler()
                            if config.preprocessing:
                                if config.use_overlapping:
                                    if config.use_last_point:
                                        validation_unroll = val_train.detach().cpu().numpy()[:, -1]
                                        validation_unroll = min_max_scaler.fit_transform(validation_unroll)
                                        validation_original = val_input[:, -1]
                                    else:
                                        validation_unroll = unroll_window_3D(np.reshape(val_train.detach().cpu().numpy(), 
                                                                                            (-1, config.rolling_size, original_x_dim)))[::-1]
                                        validation_unroll = min_max_scaler.fit_transform(validation_unroll)
                                        validation_original = unroll_window_3D(np.reshape(val_input, (-1, config.rolling_size, original_x_dim)))[::-1]
                                else:
                                    validation_unroll = np.reshape(val_train.detach().cpu().numpy(), (-1, original_x_dim))
                                    validation_unroll = min_max_scaler.fit_transform(validation_unroll)
                                    validation_original = np.reshape(val_input, (-1, original_x_dim))

                            else:
                                validation_unroll = val_train.detach().cpu().numpy()
                                validation_unroll = np.transpose(np.squeeze(validation_unroll, axis=0))
                                validation_unroll = min_max_scaler.fit_transform(validation_unroll)
                                validation_original = val_input

                            scaler = preprocessing.StandardScaler()
                            validation_original = scaler.fit_transform(validation_original)

                            validation_error = np.sum(validation_original - np.reshape(validation_unroll, [-1, original_x_dim]), axis=1) ** 2
                            error_by_timestamp = np.sum(validation_error)/validation_original.shape[0]
                            train_error = np.sum(error)/x_original_unroll.shape[0]

                            #Save results, SQL
                        
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
            cat_xs = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                batch_x = batch_x.to(device)
                batch_x_reconstruct = self.forward(batch_x)
                cat_xs.append(batch_x_reconstruct)

            if self.preprocessing:
                cat_xs = torch.cat(cat_xs, dim=0)
            else:
                cat_xs = torch.cat(cat_xs, dim=2)

            cat_train = []
            for i, (batch_x, batch_y) in enumerate(train_data):
                batch_x = batch_x.to(device)
                batch_x_reconstruct = self.forward(batch_x)
                cat_train.append(batch_x_reconstruct)

            if self.preprocessing:
                cat_train = torch.cat(cat_train, dim=0)
            else:
                cat_train = torch.cat(cat_train, dim=0)

            if self.validation:  
                valid_data = get_loader(input=val_input, label=None, batch_size=self.batch_size, from_numpy=True,
                               drop_last=False, shuffle=False)
                val_train = []
                for i, (batch_x, batch_y) in enumerate(valid_data):
                    batch_x = batch_x.to(device)
                    batch_x_reconstruct = self.forward(batch_x)
                    val_train.append(batch_x_reconstruct)

                if self.preprocessing:
                    val_train = torch.cat(val_train, dim=0)
                else:
                    val_train = torch.cat(val_train, dim=0)
                    
                cae_output = CAEOutput(dec_means=cat_xs, train_means=cat_train, validation_means=val_train,
                                       best_TN=max(TN), best_FP=max(FP),best_FN=max(FN), best_TP=max(TP),
                                       best_precision=max(PRECISION), best_recall=max(RECALL), best_fbeta=max(FBETA),
                                       best_pr_auc=max(PR_AUC), best_roc_auc=max(ROC_AUC), best_cks=max(CKS))
            else:
                cae_output = CAEOutput(dec_means=cat_xs, train_means=cat_train, validation_means=None,
                                       best_TN=max(TN), best_FP=max(FP),best_FN=max(FN), best_TP=max(TP),
                                       best_precision=max(PRECISION), best_recall=max(RECALL), best_fbeta=max(FBETA),
                                       best_pr_auc=max(PR_AUC), best_roc_auc=max(ROC_AUC), best_cks=max(CKS))
            
            execution_time[len(ensemble_member)] = time.time() - start_time_file
            
            return cae_output


# In[6]:


class CAE_Ensemble(nn.Module):
    def __init__(self, file_name, config):
        super(CAE_Ensemble, self).__init__()
        self.cae = CAE(file_name=file_name, config=config)
        self.members = config.ensemble_members
        self.beta = config.beta_transfer
        self.outputs = []

    def fit(self, train_input, val_input, test_input, test_label, abnormal_data, abnormal_label, original_x_dim):
        for member in range(self.members):
            if member > 1:
                parameters_last_model = len(list(self.cae.parameters()))
                generic_features = random.sample(range(1, parameters_last_model),
                                                 math.floor(parameters_last_model * self.beta))

                index = 0

                for parameter in self.cae.parameters():
                    index += 1
                    if index in generic_features:
                        parameter.requires_grad = True
                    else:
                        parameter.requires_grad = False

            self.outputs.append(
                self.cae.fit(train_input, val_input, test_input, test_label, abnormal_data, abnormal_label,
                             original_x_dim, self.outputs))

        return self.outputs


def spot_thread(abnormal_segment, error, config, cae_output):
    final_zscore = zscore(error)

    s = SPOT(config.spot_q)
    s.fit(final_zscore, final_zscore)
    s.initialize(level=config.spot_level)
    ret = s.run()

    spot_result = np.mean(ret['thresholds'])
    np_decision = create_label_based_on_zscore(final_zscore, spot_result, True)
    pos_label = -1

    cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
    precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
    recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
    fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
    fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), y_score=np.nan_to_num(error), pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment), probas_pred=np.nan_to_num(error),
                                        pos_label=pos_label)
    pr_auc = auc(re, pre)
    cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)
    settings = config.to_string()
    conn = sqlite3.connect('./experiments.db')
    cursor_obj = conn.cursor()
    insert_sql = """INSERT or REPLACE into model (model_name, pid, settings, dataset, file_name, TN, FP, FN, 
    TP, precision, recall, fbeta, pr_auc, roc_auc, cks, best_TN, best_FP, best_FN, best_TP, best_precision, 
    best_recall, best_fbeta, best_pr_auc, best_roc_auc, best_cks) VALUES('{}', '{}', '{}', '{}', '{}', '{}', 
    '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', 
    '{}', '{}')""".format(
        'CAE_ENSEMBLE', config.pid + 100, settings, config.dataset, Path(file_name).stem, cm[0][0], cm[0][1], cm[1][0],
        cm[1][1], precision, recall, fbeta, pr_auc, roc_auc, cks, cae_output.best_TN, cae_output.best_FP,
        cae_output.best_FN, cae_output.best_TP, cae_output.best_precision, cae_output.best_recall,
        cae_output.best_fbeta, cae_output.best_pr_auc, cae_output.best_roc_auc, cae_output.best_cks)
    cursor_obj.execute(insert_sql)
    conn.commit()
    conn.close()


# In[9]:


def RunModel(file_name, config):
    train_data = None
    if config.dataset == 0:
        abnormal_data, abnormal_label = generate_synthetic_dataset(case=0, N=512, noise=True, verbose=False)
    if config.dataset == 1:
        abnormal_data, abnormal_label = read_GD_dataset(file_name)
    if config.dataset == 2:
        abnormal_data, abnormal_label = read_HSS_dataset(file_name)
    if config.dataset == 31 or config.dataset == 32 or config.dataset == 33 or config.dataset == 34 or config.dataset == 35:
        abnormal_data, abnormal_label = read_S5_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 41 or config.dataset == 42 or config.dataset == 43 or config.dataset == 44 or config.dataset == 45 or config.dataset == 46:
        abnormal_data, abnormal_label = read_NAB_dataset(file_name)
    if config.dataset == 51 or config.dataset == 52 or config.dataset == 53 or config.dataset == 54 or config.dataset == 55 or config.dataset == 56 or config.dataset == 57:
        train_data, abnormal_data, abnormal_label = read_2D_dataset(file_name)
    if config.dataset == 61 or config.dataset == 62 or config.dataset == 63 or config.dataset == 64 or config.dataset == 65 or config.dataset == 66 or config.dataset == 67:
        abnormal_data, abnormal_label = read_ECG_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 71 or config.dataset == 72 or config.dataset == 73:
        train_data, abnormal_data, abnormal_label = read_SMD_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 81 or config.dataset == 82 or config.dataset == 83 or config.dataset == 84 or config.dataset == 85 or config.dataset == 86 or config.dataset == 87 or config.dataset == 88 or config.dataset == 89 or config.dataset == 90:
        train_data, abnormal_data, abnormal_label = read_SMAP_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 91 or config.dataset == 92 or config.dataset == 93 or config.dataset == 94 or config.dataset == 95 or config.dataset == 96 or config.dataset == 97:
        train_data, abnormal_data, abnormal_label = read_MSL_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 101:
        train_data, abnormal_data, abnormal_label = read_SWAT_dataset(file_name, sampling=0.1, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 102 or config.dataset == 103:
        train_data, abnormal_data, abnormal_label = read_WADI_dataset(file_name, sampling=0.1, include_slope = config.use_slope, slope = config.slope)

    original_x_dim = abnormal_data.shape[1]

    if config.preprocessing:
        if config.use_overlapping:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(train_data,
                                                                                                      config.rolling_size), rolling_window_2D(
                    abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
            else:
                rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(abnormal_data,
                                                                                  config.rolling_size), rolling_window_2D(
                    abnormal_label, config.rolling_size)
        else:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(train_data,
                                                                                                      config.rolling_size), cutting_window_2D(
                    abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
            else:
                rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(abnormal_data,
                                                                                  config.rolling_size), cutting_window_2D(
                    abnormal_label, config.rolling_size)
               
        config.x_dim = rolling_abnormal_data.shape[2]
    else:
        if train_data is not None:
            rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = train_data, abnormal_data, abnormal_label
        else:
            rolling_abnormal_data, rolling_abnormal_label = abnormal_data, abnormal_label

        config.x_dim = rolling_abnormal_data.shape[1]

    if config.ensemble:
        model = CAE_Ensemble(file_name=file_name, config=config)
    else:
        model = CAE(file_name=file_name, config=config)

    model = model.to(device)

    if train_data is not None and config.robustness == False:
        cae_output = model.fit(train_input=rolling_train_data, val_input=rolling_train_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                               original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        cae_output = model.fit(train_input=rolling_abnormal_data, val_input=rolling_abnormal_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                               original_x_dim=original_x_dim)

    if config.ensemble:
        cae_outputs = cae_output
    else:
        cae_outputs = []
        cae_outputs.append(cae_output)
    
    for ensemble_index, cae_output in enumerate(cae_outputs):
        min_max_scaler = preprocessing.StandardScaler()
        if config.preprocessing:
            if config.use_overlapping:
                if config.use_last_point:
                    dec_mean_unroll = cae_output.dec_means.detach().cpu().numpy()[:, -1]
                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                    x_original_unroll = abnormal_data[config.rolling_size-1:]
                    abnormal_segment = abnormal_label[config.rolling_size-1:]
                else:
                    dec_mean_unroll = unroll_window_3D(
                        np.reshape(cae_output.dec_means.detach().cpu().numpy(), (-1, config.rolling_size, original_x_dim)))[
                                      ::-1]
                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                    x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                    abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
            else:
                dec_mean_unroll = np.reshape(cae_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
        else:
            dec_mean_unroll = cae_output.dec_means.detach().cpu().numpy()
            dec_mean_unroll = np.transpose(np.squeeze(dec_mean_unroll, axis=0))
            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            x_original_unroll = abnormal_data
            abnormal_segment = abnormal_label

        model_name = 'CAE_ENSEMBLE_' + str(ensemble_index)
        if config.save_output:
            if not os.path.exists('./save_outputs/NPY/{}/'.format(config.dataset)):
                os.makedirs('./save_outputs/NPY/{}/'.format(config.dataset))
            np.save('./save_outputs/NPY/{}/{}#{}#{}#{}.npy'.format(config.dataset, model_name,
                                                           Path(file_name).stem,config.pid,config.rolling_size),dec_mean_unroll)

        scaler = preprocessing.StandardScaler()
        x_original_unroll = scaler.fit_transform(x_original_unroll)

        error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2

        # %%
        if config.save_figure:
            if not os.path.exists('./save_figures/{}/'.format(config.dataset)):
                os.makedirs('./save_figures/{}/'.format(config.dataset))
            if original_x_dim == 1:
                plt.figure(figsize=(9, 3))
                plt.plot(x_original_unroll, color='blue', lw=1.5)
                plt.plot(dec_mean_unroll)
                plt.title('Original Data')
                plt.grid(True)
                plt.tight_layout()
                #plt.show()
                plt.savefig('./save_figures/{}/Ori_CAE_ENSEMBLE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
                plt.close()

                # Plot decoder output
                plt.figure(figsize=(9, 3))
                plt.plot(dec_mean_unroll, color='blue', lw=1.5)
                plt.title('Decoding Output')
                plt.grid(True)
                plt.tight_layout()
                #plt.show()
                plt.savefig('./save_figures/{}/Dec_CAE_ENSEMBLE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
                plt.close()

                t = np.arange(0, abnormal_data.shape[0])
                markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in abnormal_label[: dec_mean_unroll.shape[0]]]
                markersize = [4 for i in range(config.rolling_size - 1)] + [4 if i == 1 else 25 for i in abnormal_label[: dec_mean_unroll.shape[0]]]
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
                plt.scatter(t, x_original_unroll, s=markersize, c=markercolors)
                # plt.show()
                plt.savefig('./save_figures/{}/VisInp_CAE_ENSEMBLE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
                plt.close()

                markercolors = ['blue' for i in range(config.rolling_size - 1)] + ['blue' if i == 1 else 'red' for i in np_decision]
                markersize = [4 for i in range(config.rolling_size - 1)] + [4 if i == 1 else 25 for i in np_decision]
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
                plt.savefig('./save_figures/{}/VisOut_CAE_ENSEMBLE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
                plt.close()
            else:
                file_logger.info('cannot plot image with x_dim > 1')

        try:
            if config.use_spot:
                spot_thread(abnormal_segment, error, config, cae_output)

            pos_label = -1
            score = error   
            
            std_dev_values = [2,2.5,3]
            quantile_values = [97,98,99]
            k_values = [0.5,1,2,4,5,10]

            CalculateMetrics(abnormal_segment, score, pos_label, std_dev_values, quantile_values, k_values, model_name, int(config.pid), int(config.dataset), Path(file_name).stem)
            
            final_zscore = zscore(error)
            np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
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
            
            metrics_result = MetricsResult(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1], precision=precision,
                                           recall=recall, fbeta=fbeta, pr_auc=pr_auc, roc_auc=roc_auc, cks=cks,
                                           best_TN=cae_output.best_TN, best_FP=cae_output.best_FP,
                                           best_FN=cae_output.best_FN, best_TP=cae_output.best_TP,
                                           best_precision=cae_output.best_precision, best_recall=cae_output.best_recall,
                                           best_fbeta=cae_output.best_fbeta, best_pr_auc=cae_output.best_pr_auc,
                                           best_roc_auc=cae_output.best_roc_auc, best_cks=cae_output.best_cks)
            if (ensemble_index + 1) == len(cae_outputs):
                return metrics_result

        except Exception as e:
            print(e)


# In[10]:


def RunValidation(file_name, config):
    cursor_obj.execute('''CREATE TABLE IF NOT EXISTS validation
             (model_name, pid, settings, dataset, file_name, TN, FP, FN, TP, precision, 
             recall, fbeta, pr_auc, roc_auc, cks, best_TN, best_FP, best_FN, best_TP, 
             best_precision, best_recall, best_fbeta, best_pr_auc, best_roc_auc, best_cks, validation_error, 
             diversity_factor, beta_transfer)''')
    conn.commit()
    train_data = None
    if config.dataset == 31 or config.dataset == 32 or config.dataset == 33 or config.dataset == 34 or config.dataset == 35:
        abnormal_data, abnormal_label = read_S5_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 41 or config.dataset == 42 or config.dataset == 43 or config.dataset == 44 or config.dataset == 45 or config.dataset == 46:
        abnormal_data, abnormal_label = read_NAB_dataset(file_name)
    if config.dataset == 51 or config.dataset == 52 or config.dataset == 53 or config.dataset == 54 or config.dataset == 55 or config.dataset == 56 or config.dataset == 57:
        train_data, abnormal_data, abnormal_label = read_2D_dataset(file_name)
    if config.dataset == 61 or config.dataset == 62 or config.dataset == 63 or config.dataset == 64 or config.dataset == 65 or config.dataset == 66 or config.dataset == 67:
        abnormal_data, abnormal_label = read_ECG_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 71 or config.dataset == 72 or config.dataset == 73:
        train_data, abnormal_data, abnormal_label = read_SMD_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 81 or config.dataset == 82 or config.dataset == 83 or config.dataset == 84 or config.dataset == 85 or config.dataset == 86 or config.dataset == 87 or config.dataset == 88 or config.dataset == 89 or config.dataset == 90:
        train_data, abnormal_data, abnormal_label = read_SMAP_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 91 or config.dataset == 92 or config.dataset == 93 or config.dataset == 94 or config.dataset == 95 or config.dataset == 96 or config.dataset == 97:
        train_data, abnormal_data, abnormal_label = read_MSL_dataset(file_name, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 101:
        train_data, abnormal_data, abnormal_label = read_SWAT_dataset(file_name, sampling=0.1, include_slope = config.use_slope, slope = config.slope)
    if config.dataset == 102 or config.dataset == 103:
        train_data, abnormal_data, abnormal_label = read_WADI_dataset(file_name, sampling=0.1, include_slope = config.use_slope, slope = config.slope)

    original_x_dim = abnormal_data.shape[1]

    if config.preprocessing:
        if config.use_overlapping:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(train_data,
                                                                                                      config.rolling_size), rolling_window_2D(
                    abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
            else:
                rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(abnormal_data,
                                                                                  config.rolling_size), rolling_window_2D(
                    abnormal_label, config.rolling_size)
        else:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(train_data,
                                                                                                      config.rolling_size), cutting_window_2D(
                    abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
            else:
                rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(abnormal_data,
                                                                                  config.rolling_size), cutting_window_2D(
                    abnormal_label, config.rolling_size)

        config.x_dim = rolling_abnormal_data.shape[2]
    else:
        if train_data is not None:
            rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = train_data, abnormal_data, abnormal_label
        else:
            rolling_abnormal_data, rolling_abnormal_label = abnormal_data, abnormal_label

        config.x_dim = rolling_abnormal_data.shape[1]
        
    if config.ensemble:
        model = CAE_Ensemble(file_name=file_name, config=config)
    else:
        model = CAE(file_name=file_name, config=config)

    model = model.to(device)

    if train_data is not None and config.robustness == False:
        rolling_train_data, validation_data = np.split(rolling_train_data, [int(0.7 * len(rolling_train_data))])
        cae_output = model.fit(train_input=rolling_train_data, val_input=validation_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                               original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        rolling_abnormal_data_cut, validation_data = np.split(rolling_abnormal_data, [int(0.7 * len(rolling_abnormal_data))])
        cae_output = model.fit(train_input=rolling_abnormal_data_cut, val_input=validation_data,
                               test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                               abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                               original_x_dim=original_x_dim)
    
    if config.ensemble:
        cae_outputs = cae_output
    else:
        cae_outputs = []
        cae_outputs.append(cae_output)

    for ensemble_index, cae_output in enumerate(cae_outputs):
        min_max_scaler = preprocessing.StandardScaler()
        if config.preprocessing:
            if config.use_overlapping:
                if config.use_last_point:
                    dec_mean_unroll = cae_output.dec_means.detach().cpu().numpy()[:, -1]
                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                    x_original_unroll = abnormal_data[config.rolling_size-1:]
                    abnormal_segment = abnormal_label[config.rolling_size-1:]

                    validation_unroll = cae_output.validation_means.detach().cpu().numpy()[:, -1]
                    validation_unroll = min_max_scaler.fit_transform(validation_unroll)
                    validation_original = validation_data[:, -1]
                else:
                    dec_mean_unroll = unroll_window_3D(
                        np.reshape(cae_output.dec_means.detach().cpu().numpy(),
                                   (-1, config.rolling_size, original_x_dim)))[::-1]
                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                    x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                    abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]

                    validation_unroll = unroll_window_3D(np.reshape(cae_output.validation_means.detach().cpu().numpy(), 
                                                                    (-1, config.rolling_size, original_x_dim)))[::-1]
                    validation_unroll = min_max_scaler.fit_transform(validation_unroll)
                    validation_original = unroll_window_3D(np.reshape(validation_data, (-1, config.rolling_size, original_x_dim)))[::-1]
            else:
                dec_mean_unroll = np.reshape(cae_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]

                validation_unroll = np.reshape(cae_output.validation_means.detach().cpu().numpy(), (-1, original_x_dim))
                validation_unroll = min_max_scaler.fit_transform(validation_unroll)
                validation_original = np.reshape(validation_data, (-1, original_x_dim))

        else:
            dec_mean_unroll = cae_output.dec_means.detach().cpu().numpy()
            dec_mean_unroll = np.transpose(np.squeeze(dec_mean_unroll, axis=0))
            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            x_original_unroll = abnormal_data
            abnormal_segment = abnormal_label

            validation_unroll = cae_output.validation_means.detach().cpu().numpy()
            validation_unroll = np.transpose(np.squeeze(validation_unroll, axis=0))
            validation_unroll = min_max_scaler.fit_transform(validation_unroll)
            validation_original = validation_data

        scaler = preprocessing.StandardScaler()
        x_original_unroll = scaler.fit_transform(x_original_unroll)
        validation_original = scaler.fit_transform(validation_original)

        error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2
        validation_error = np.sum(validation_original - np.reshape(validation_unroll, [-1, original_x_dim]), axis=1) ** 2
        error_by_timestamp = np.sum(validation_error)/validation_original.shape[0]
        train_error = np.sum(error)/x_original_unroll.shape[0]        
        
        pos_label = -1
        score = error   

        std_dev_values = [2,2.5,3]
        quantile_values = [97,98,99]
        k_values = [0.5,1,2,4,5,10]
        
        model_name = 'CAE_ENSEMBLE_' + str(ensemble_index)

        ValidationMetrics(abnormal_segment, score, pos_label, std_dev_values, quantile_values, k_values, model_name, 
                          int(config.pid), int(config.dataset), Path(file_name).stem, train_error, error_by_timestamp,
                          config.beta_transfer, config.diversity_factor, config.rolling_size)

        final_zscore = zscore(error)
        np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)
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

        metrics_result = MetricsResult(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1], precision=precision,
                                       recall=recall, fbeta=fbeta, pr_auc=pr_auc, roc_auc=roc_auc, cks=cks,
                                       best_TN=cae_output.best_TN, best_FP=cae_output.best_FP,
                                       best_FN=cae_output.best_FN, best_TP=cae_output.best_TP,
                                       best_precision=cae_output.best_precision, best_recall=cae_output.best_recall,
                                       best_fbeta=cae_output.best_fbeta, best_pr_auc=cae_output.best_pr_auc,
                                       best_roc_auc=cae_output.best_roc_auc, best_cks=cae_output.best_cks)
        if (ensemble_index + 1) == len(cae_outputs):
            return metrics_result

# In[11]:


if __name__ == '__main__':
    conn = sqlite3.connect('./experiments.db')
    cursor_obj = conn.cursor()

    cursor_obj.execute('''CREATE TABLE IF NOT EXISTS results
             (model_name, pid, settings, dataset, file_name, TN, FP, FN, TP, precision, 
             recall, fbeta, pr_auc, roc_auc, cks, best_TN, best_FP, best_FN, best_TP, 
             best_precision, best_recall, best_fbeta, best_pr_auc, best_roc_auc, best_cks, time)''')
    conn.commit()    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fff", help="a dummy argument for Jupyter", default="1")
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    parser.add_argument('--rolling_size', type=int, default=32)
    parser.add_argument('--use_best_f1', type=str2bool, default=False)
    parser.add_argument('--ensemble', type=str2bool, default=True)
    parser.add_argument('--ensemble_members', type=int, default=8)
    parser.add_argument('--beta_transfer', type=float, default=0.8)
    parser.add_argument('--diversity_factor', type=float, default=0.7)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--early_stopping', type=str2bool, default=False)
    parser.add_argument('--loss_function', type=str, default='mse')
    parser.add_argument('--display_epoch', type=int, default=5)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=False)
    parser.add_argument('--save_model', type=str2bool, default=False)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--continue_training', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--spot_q', type=float, default=0.01)
    parser.add_argument('--spot_level', type=float, default=0.04)
    parser.add_argument('--use_last_point', type=str2bool, default=True)
    parser.add_argument('--adjusted_points', type=str2bool, default=False)
    parser.add_argument('--use_slope', type=str2bool, default=False)
    parser.add_argument('--slope', type=float, default=0.01)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--use_clip_norm', type=str2bool, default=True)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--validation', type=int, default=0)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    if args.load_config:
        config = CAEConfig(dataset=None, x_dim=None, h_dim=None, preprocessing=None, use_overlapping=None,
                           rolling_size=None, ensemble=None, ensemble_members=None, beta_transfer=None,
                           diversity_factor=None, validation=None, use_clip_norm=None, gradient_clip_norm=None,
                           epochs=None, milestone_epochs=None, lr=None, gamma=None, batch_size=None, use_best_f1=None,
                           weight_decay=None, early_stopping=None, loss_function=None, display_epoch=None,
                           save_output=None, save_figure=None, save_model=None, load_model=None, continue_training=None,
                           dropout=None, use_spot=None, spot_q=None, spot_level=None, use_last_point=None,
                           save_config=None, use_slope=None, slope=None,
                           adjusted_points=None, load_config=None, server_run=None, robustness=None, pid=None)
        try:
            config.import_config('./save_configs/{}/Config_CAE_ENSEMBLE_pid={}.json'.format(config.dataset, config.pid))
        except:
            print('There is no config.')
    else:
        config = CAEConfig(dataset=args.dataset, x_dim=args.x_dim, h_dim=args.h_dim, preprocessing=args.preprocessing,
                           use_overlapping=args.use_overlapping, rolling_size=args.rolling_size, ensemble=args.ensemble,
                           ensemble_members=args.ensemble_members, beta_transfer=args.beta_transfer,
                           diversity_factor=args.diversity_factor,
                           epochs=args.epochs, milestone_epochs=args.milestone_epochs, lr=args.lr, gamma=args.gamma,
                           batch_size=args.batch_size, weight_decay=args.weight_decay, spot_q=args.spot_q,
                           spot_level=args.spot_level, validation=args.validation,
                           early_stopping=args.early_stopping, loss_function=args.loss_function,
                           use_best_f1=args.use_best_f1, use_slope=args.use_slope, slope=args.slope,
                           display_epoch=args.display_epoch, save_output=args.save_output, save_figure=args.save_figure,
                           save_model=args.save_model, load_model=args.load_model, adjusted_points=args.adjusted_points,
                           continue_training=args.continue_training, dropout=args.dropout, use_spot=args.use_spot,
                           use_last_point=args.use_last_point, save_config=args.save_config,
                           load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                           pid=args.pid, use_clip_norm=args.use_clip_norm, gradient_clip_norm=args.gradient_clip_norm)
    if args.save_config:
        if not os.path.exists('./save_configs/{}/'.format(config.dataset)):
            os.makedirs('./save_configs/{}/'.format(config.dataset))
        config.export_config('./save_configs/{}/Config_CAE_ENSEMBLE_pid={}.json'.format(config.dataset, config.pid))
    # %%
    device = torch.device(get_free_device())
    train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                           train_logger_name='cae_ensemble_train_logger',
                                                           file_logger_name='cae_ensemble_file_logger',
                                                           meta_logger_name='cae_ensemble_meta_logger',
                                                           model_name='CAE_ENSEMBLE',
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
                s_TN, s_FP, s_FN, s_TP, s_precision, s_recall, s_fbeta, s_roc_auc, s_pr_auc, s_cks = ([] for _ in
                                                                                                      range(10))
                s_best_TN, s_best_FP, s_best_FN, s_best_TP, s_best_precision = ([] for _ in range(5))
                s_best_recall, s_best_fbeta, s_best_roc_auc, s_best_pr_auc, s_best_cks = ([] for _ in range(5))

                for file in files:
                    file_name = os.path.join(root, file)
                    file_logger.info('============================')
                    file_logger.info(file)
                    
                    start_time_file = time.time()
                    execution_time = np.empty(config.ensemble_members)
                    
                    if config.validation == 1: # Beta, lambda, windows size
                        for beta_validation in [float(j) / 10 for j in range(1, 10, 1)]:
                            config.beta_transfer = beta_validation
                            for lambda_validation in (1,2,4,8,16,32,64):
                                config.diversity_factor = lambda_validation
                                for windows_validation in (4,8,16,32,64,128,256):
                                    config.rolling_size = windows_validation
                                    metrics_result = RunValidation(file_name=file_name, config=config)
                    elif config.validation == 3: # Epochs
                        metrics_result = RunValidation(file_name=file_name, config=config)
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
                    s_best_TN.append(metrics_result.best_TN)
                    file_logger.info('best_TN = {}'.format(metrics_result.best_TN))
                    s_best_FP.append(metrics_result.best_FP)
                    file_logger.info('best_FP = {}'.format(metrics_result.best_FP))
                    s_best_FN.append(metrics_result.best_FN)
                    file_logger.info('best_FN = {}'.format(metrics_result.best_FN))
                    s_best_TP.append(metrics_result.best_TP)
                    file_logger.info('best_TP = {}'.format(metrics_result.best_TP))
                    s_best_precision.append(metrics_result.best_precision)
                    file_logger.info('best_precision = {}'.format(metrics_result.best_precision))
                    s_best_recall.append(metrics_result.best_recall)
                    file_logger.info('best_recall = {}'.format(metrics_result.best_recall))
                    s_best_fbeta.append(metrics_result.best_fbeta)
                    file_logger.info('best_fbeta = {}'.format(metrics_result.best_fbeta))
                    s_best_roc_auc.append(metrics_result.best_roc_auc)
                    file_logger.info('best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                    s_best_pr_auc.append(metrics_result.best_pr_auc)
                    file_logger.info('best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                    s_best_cks.append(metrics_result.best_cks)
                    file_logger.info('best_cks = {}'.format(metrics_result.best_cks))

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
                avg_best_TN = calculate_average_metric(s_best_TN)
                meta_logger.info('avg_best_TN = {}'.format(avg_best_TN))
                avg_best_FP = calculate_average_metric(s_best_FP)
                meta_logger.info('avg_best_FP = {}'.format(avg_best_FP))
                avg_best_FN = calculate_average_metric(s_best_FN)
                meta_logger.info('avg_best_FN = {}'.format(avg_best_FN))
                avg_best_TP = calculate_average_metric(s_best_TP)
                meta_logger.info('avg_best_TP = {}'.format(avg_best_TP))
                avg_best_precision = calculate_average_metric(s_best_precision)
                meta_logger.info('avg_best_precision = {}'.format(avg_best_precision))
                avg_best_recall = calculate_average_metric(s_best_recall)
                meta_logger.info('avg_best_recall = {}'.format(avg_best_recall))
                avg_best_fbeta = calculate_average_metric(s_best_fbeta)
                meta_logger.info('avg_best_fbeta = {}'.format(avg_best_fbeta))
                avg_best_roc_auc = calculate_average_metric(s_best_roc_auc)
                meta_logger.info('avg_best_roc_auc = {}'.format(avg_best_roc_auc))
                avg_best_pr_auc = calculate_average_metric(s_best_pr_auc)
                meta_logger.info('avg_best_pr_auc = {}'.format(avg_best_pr_auc))
                avg_best_cks = calculate_average_metric(s_best_cks)
                meta_logger.info('avg_best_cks = {}'.format(avg_best_cks))
                file_logger.info('Finish')
                meta_logger.info('Finish')
                conn.close()

    if args.server_run:
        send_email_notification(
            subject='baseline_cae_ensemble application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid),
            message='baseline_cae_ensemble application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid))