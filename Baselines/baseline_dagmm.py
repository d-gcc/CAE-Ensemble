import argparse
import os
import sqlite3
import traceback

import matplotlib as mpl
from sklearn import preprocessing
from torch.autograd import Variable
from utils.device import get_free_device
from utils.mail import send_email_notification
from utils.outputs import DAGMMOutput
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from statistics import mean
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, auc, precision_recall_curve, roc_curve, fbeta_score, confusion_matrix
from torch.optim import Adam, lr_scheduler
from utils.config import DAGMMConfig
from utils.data_provider import read_GD_dataset, generate_synthetic_dataset, read_HSS_dataset, read_S5_dataset, \
    read_NAB_dataset, read_2D_dataset, read_ECG_dataset, read_SMD_dataset, rolling_window_2D, \
    get_loader, cutting_window_2D, unroll_window_3D, read_SMAP_dataset, read_MSL_dataset, unroll_window_2D, \
    read_WADI_dataset, read_SWAT_dataset
from utils.logger import create_logger
from utils.metrics import calculate_average_metric, zscore, create_label_based_on_zscore, \
    MetricsResult
from utils.utils import str2bool


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s


class DAGMM(nn.Module):
    def __init__(self, file_name, config):
        """Network for DAGMM (KDDCup99)"""
        super(DAGMM, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # dim info
        self.x_dim = config.x_dim
        self.h_dim = config.h_dim
        self.z_dim = config.z_dim
        self.n_gmm = config.n_gmm

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
        self.display_epoch = config.display_epoch
        self.lambda_energy = config.lambda_energy
        self.lambda_cov = config.lambda_cov
        self.use_clip_norm = config.use_clip_norm
        self.gradient_clip_norm = config.gradient_clip_norm

        # dropout
        self.dropout = config.dropout
        self.continue_training = config.continue_training

        self.robustness = config.robustness

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim * 6),
            nn.Tanh(),
            nn.Linear(self.h_dim * 6, self.h_dim * 3),
            nn.Tanh(),
            nn.Linear(self.h_dim * 3, self.h_dim),
            nn.Tanh(),
            nn.Linear(self.h_dim, self.z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.Tanh(),
            nn.Linear(self.h_dim, self.h_dim * 3),
            nn.Tanh(),
            nn.Linear(self.h_dim * 3, self.h_dim * 6),
            nn.Tanh(),
            nn.Linear(self.h_dim * 6, self.x_dim)
        )

        self.estimator = nn.Sequential(
            nn.Linear(self.z_dim + 2, self.h_dim),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(self.h_dim, self.n_gmm),
            nn.Softmax()
        )

        self.pid = config.pid

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_models/{}/'.format(self.dataset)):
                os.makedirs('./save_models/{}/'.format(self.dataset))
            self.save_model_path = \
                './save_models/{}/DAGMM' \
                '_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_models/{}/DAGMM' \
                '_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = nn.functional.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        #  z = NxD
        # gamma = NxK

        # phi = D
        phi = torch.sum(gamma, dim=0) / gamma.size(0)

        # mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov

    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1)) * eps).to(device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean == True:
            E_z = torch.mean(E_z)
        return E_z, cov_diag

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([h, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimator(z)
        return h, x_hat, z, gamma

    def compute_loss(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x - x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)

    def fit(self, train_input, train_label, test_input, test_label, abnormal_data, abnormal_label, original_x_dim):
        TN = []
        TP = []
        FN = []
        FP = []
        PRECISION = []
        RECALL = []
        FBETA = []
        PR_AUC = []
        ROC_AUC = []
        CKS = []
        opt = Adam(list(self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)
        # get batch data
        train_data = get_loader(input=train_input, label=train_label, batch_size=self.batch_size, from_numpy=True,
                                drop_last=False, shuffle=True)
        test_data = get_loader(input=test_input, label=test_label, batch_size=self.batch_size, from_numpy=True,
                               drop_last=False, shuffle=False)
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
                    # opt.zero_grad()
                    batch_x = batch_x.to(device)
                    _, batch_x_hat, batch_z, batch_gamma = self.forward(batch_x)
                    batch_loss = self.compute_loss(x=batch_x, x_hat=batch_x_hat, z=batch_z, gamma=batch_gamma)
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
                        N_samples = 0
                        gamma_sum = 0
                        mu_sum = 0
                        cov_sum = 0
                        # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
                        for i, (batch_x, batch_y) in enumerate(train_data):
                            batch_x = batch_x.to(device)
                            _, _, batch_z, batch_gamma = self.forward(batch_x)
                            phi_batch, mu_batch, cov_batch = self.compute_params(batch_z, batch_gamma)

                            batch_gamma_sum = torch.sum(batch_gamma, dim=0)
                            gamma_sum += batch_gamma_sum
                            mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
                            cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)

                            N_samples += batch_x.size(0)

                        train_phi = gamma_sum / N_samples
                        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
                        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)
                        energy = []
                        x_hat = []
                        for i, (batch_x, batch_y) in enumerate(test_data):
                            batch_x = batch_x.to(device)
                            _, batch_x_hat, batch_z, batch_gamma = self.forward(batch_x)
                            sample_energy, cov_diag = self.compute_energy(batch_z, batch_gamma, phi=train_phi,
                                                                          mu=train_mu, cov=train_cov, sample_mean=False)
                            energy.append(sample_energy)
                            x_hat.append(batch_x_hat)
                        energy = torch.cat(energy)
                        x_hat = torch.cat(x_hat)
                        dagmm_output = DAGMMOutput(dec_means=x_hat, energy=energy, best_TN=None, best_FP=None,
                                                   best_FN=None, best_TP=None, best_precision=None, best_recall=None,
                                                   best_fbeta=None, best_pr_auc=None, best_roc_auc=None, best_cks=None)

                        min_max_scaler = preprocessing.MinMaxScaler()
                        if self.preprocessing:
                            if self.use_overlapping:
                                dec_mean_unroll = unroll_window_3D(np.reshape(dagmm_output.dec_means.detach().cpu().numpy(), (-1, self.rolling_size, original_x_dim)))[::-1]
                                energy_unroll = dagmm_output.energy.detach().cpu().numpy()
                                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]

                            else:
                                dec_mean_unroll = np.reshape(dagmm_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
                                energy_unroll = dagmm_output.energy.detach().cpu().numpy()
                                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                        else:
                            dec_mean_unroll = dagmm_output.dec_means.detach().cpu().numpy()
                            energy_unroll = dagmm_output.energy.detach().cpu().numpy()
                            x_original_unroll = abnormal_data

                        error = np.sum(abnormal_data[: dec_mean_unroll.shape[0]] - np.reshape(dec_mean_unroll, [-1, original_x_dim]),
                                       axis=1) ** 2
                        final_zscore = zscore(energy_unroll)
                        np_decision = create_label_based_on_zscore(final_zscore, 3, True)

                        pos_label = -1
                        abnormal_segment = abnormal_label[0:np_decision.shape[0]]

                        if config.adjusted_points:
                            label_groups = np.argwhere(np.diff(abnormal_segment.squeeze()))
                            if len(label_groups)%2 != 0:
                                label_groups = np.append(label_groups,len(abnormal_segment))

                            anomaly_groups = label_groups.squeeze().reshape(-1, 2)

                            predicted = pd.DataFrame(np_decision, columns=['Score'])

                            for segment in anomaly_groups:
                                predicted_segment = predicted.iloc[segment[0]+1:segment[1]+1]
                                try:
                                    if predicted_segment['Score'].value_counts()[1] != segment[1] - segment[0]:
                                        predicted.iloc[segment[0]+1:segment[1]+1] = -1
                                except:
                                    pass

                            np_decision = predicted.to_numpy().squeeze()

                        cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
                        pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment), probas_pred=np.nan_to_num(energy_unroll), pos_label=pos_label)
                        pr_auc = auc(re, pre)
                        fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), y_score=np.nan_to_num(energy_unroll), pos_label=pos_label)
                        roc_auc = auc(fpr, tpr)       
                        cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)

                        if config.use_best_f1:
                            metrics = pd.DataFrame(data=pre,columns=['Precision'])
                            metrics['Recall'] = re
                            metrics['F1'] = metrics.apply(lambda row: 2 * (row.Precision * row.Recall)/(row.Precision + row.Recall + 0.0000001), axis = 1) 
                            top_f1 = metrics.apply(pd.to_numeric).nlargest(1, ['F1'])

                            precision = top_f1['Precision'].values[0]
                            recall = top_f1['Recall'].values[0]
                            fbeta = top_f1['F1'].values[0]
                        else:
                            precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                            recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                            fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
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
        else:
            # train model
            self.train()
            epoch_losses = []
            for epoch in range(self.epochs):
                train_losses = []
                opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # opt.zero_grad()
                    batch_x = batch_x.to(device)
                    _, batch_x_hat, batch_z, batch_gamma = self.forward(batch_x)
                    batch_loss = self.compute_loss(x=batch_x, x_hat=batch_x_hat, z=batch_z, gamma=batch_gamma)
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
                        N_samples = 0
                        gamma_sum = 0
                        mu_sum = 0
                        cov_sum = 0
                        # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
                        for i, (batch_x, batch_y) in enumerate(train_data):
                            batch_x = batch_x.to(device)
                            _, _, batch_z, batch_gamma = self.forward(batch_x)
                            phi_batch, mu_batch, cov_batch = self.compute_params(batch_z, batch_gamma)

                            batch_gamma_sum = torch.sum(batch_gamma, dim=0)
                            gamma_sum += batch_gamma_sum
                            mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
                            cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)

                            N_samples += batch_x.size(0)

                        train_phi = gamma_sum / N_samples
                        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
                        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)
                        energy = []
                        x_hat = []
                        for i, (batch_x, batch_y) in enumerate(test_data):
                            batch_x = batch_x.to(device)
                            _, batch_x_hat, batch_z, batch_gamma = self.forward(batch_x)
                            sample_energy, cov_diag = self.compute_energy(batch_z, batch_gamma, phi=train_phi, mu=train_mu, cov=train_cov, sample_mean=False)
                            energy.append(sample_energy)
                            x_hat.append(batch_x_hat)
                        energy = torch.cat(energy)
                        x_hat = torch.cat(x_hat)
                        dagmm_output = DAGMMOutput(dec_means=x_hat, energy=energy, best_TN=None, best_FP=None,
                                                   best_FN=None, best_TP=None, best_precision=None, best_recall=None,
                                                   best_fbeta=None, best_pr_auc=None, best_roc_auc=None, best_cks=None)

                        min_max_scaler = preprocessing.MinMaxScaler()
                        if self.preprocessing:
                            if self.use_overlapping:
                                dec_mean_unroll = unroll_window_3D(np.reshape(dagmm_output.dec_means.detach().cpu().numpy(), (-1, self.rolling_size, original_x_dim)))[::-1]
                                energy_unroll = dagmm_output.energy.detach().cpu().numpy()
                                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]

                            else:
                                dec_mean_unroll = np.reshape(dagmm_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
                                energy_unroll = dagmm_output.energy.detach().cpu().numpy()
                                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                        else:
                            dec_mean_unroll = dagmm_output.dec_means.detach().cpu().numpy()
                            energy_unroll = dagmm_output.energy.detach().cpu().numpy()
                            x_original_unroll = abnormal_data

                        error = np.sum(abnormal_data[: dec_mean_unroll.shape[0]] - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2
                        final_zscore = zscore(energy_unroll)
                        np_decision = create_label_based_on_zscore(final_zscore, 3, True)

                        pos_label = -1
                        abnormal_segment = abnormal_label[0:np_decision.shape[0]]

                        if config.adjusted_points:
                            label_groups = np.argwhere(np.diff(abnormal_segment.squeeze()))
                            if (len(label_groups)%2 != 0):
                                label_groups = np.append(label_groups,len(abnormal_segment))

                            anomaly_groups = label_groups.squeeze().reshape(-1, 2)

                            predicted = pd.DataFrame(np_decision, columns=['Score'])

                            for segment in anomaly_groups:
                                predicted_segment = predicted.iloc[segment[0]+1:segment[1]+1]
                                try:
                                    if (predicted_segment['Score'].value_counts()[1] != segment[1] - segment[0]):
                                        predicted.iloc[segment[0]+1:segment[1]+1] = -1
                                except:
                                    pass

                            np_decision = predicted.to_numpy().squeeze()
    
                        cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
                        pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment), probas_pred=np.nan_to_num(energy_unroll), pos_label=pos_label)
                        pr_auc = auc(re, pre)
                        fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), y_score=np.nan_to_num(energy_unroll), pos_label=pos_label)
                        roc_auc = auc(fpr, tpr)       
                        cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)

                        if config.use_best_f1:
                            metrics = pd.DataFrame(data=pre,columns=['Precision'])
                            metrics['Recall'] = re
                            metrics['F1'] = metrics.apply(lambda row: 2 * (row.Precision * row.Recall)/(row.Precision + row.Recall + 0.0000001), axis = 1) 
                            top_f1 = metrics.apply(pd.to_numeric).nlargest(1, ['F1'])

                            precision = top_f1['Precision'].values[0]
                            recall = top_f1['Recall'].values[0]
                            fbeta = top_f1['F1'].values[0]
                        else:
                            precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                            recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                            fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
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
            N_samples = 0
            gamma_sum = 0
            mu_sum = 0
            cov_sum = 0
            # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
            for i, (batch_x, batch_y) in enumerate(train_data):
                batch_x = batch_x.to(device)
                _, _, batch_z, batch_gamma = self.forward(batch_x)
                phi_batch, mu_batch, cov_batch = self.compute_params(batch_z, batch_gamma)

                batch_gamma_sum = torch.sum(batch_gamma, dim=0)
                gamma_sum += batch_gamma_sum
                mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
                cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)

                N_samples += batch_x.size(0)

            train_phi = gamma_sum / N_samples
            train_mu = mu_sum / gamma_sum.unsqueeze(-1)
            train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)
            energy = []
            x_hat = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                batch_x = batch_x.to(device)
                _, batch_x_hat, batch_z, batch_gamma = self.forward(batch_x)
                sample_energy, cov_diag = self.compute_energy(batch_z, batch_gamma, phi=train_phi, mu=train_mu, cov=train_cov, sample_mean=False)
                energy.append(sample_energy)
                x_hat.append(batch_x_hat)
            energy = torch.cat(energy)
            x_hat = torch.cat(x_hat)
            dagmm_output = DAGMMOutput(dec_means=x_hat, energy=energy, best_TN=max(TN), best_FP=max(FP),
                                       best_FN=max(FN), best_TP=max(TP), best_precision=max(PRECISION),
                                       best_recall=max(RECALL), best_fbeta=max(FBETA), best_pr_auc=max(PR_AUC),
                                       best_roc_auc=max(ROC_AUC), best_cks=max(CKS))
            return dagmm_output


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
        train_data, abnormal_data, abnormal_label = read_SWAT_dataset(file_name)
    if config.dataset == 102 or config.dataset == 103:
        train_data, abnormal_data, abnormal_label = read_WADI_dataset(file_name)
        
    original_x_dim = abnormal_data.shape[1]

    if config.preprocessing:
        if config.use_overlapping:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(train_data, config.rolling_size), rolling_window_2D(abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
            else:
                rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
        else:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(train_data, config.rolling_size), cutting_window_2D(abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
            else:
                rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
        if train_data is not None:
            rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = np.reshape(rolling_train_data, [rolling_train_data.shape[0], rolling_train_data.shape[1] * rolling_train_data.shape[2]]), np.reshape(rolling_abnormal_data, [rolling_abnormal_data.shape[0], rolling_abnormal_data.shape[1] * rolling_abnormal_data.shape[2]]), np.reshape(rolling_abnormal_label, [rolling_abnormal_label.shape[0], rolling_abnormal_label.shape[1] * rolling_abnormal_label.shape[2]])
        else:
            rolling_abnormal_data, rolling_abnormal_label = np.reshape(rolling_abnormal_data, [rolling_abnormal_data.shape[0], rolling_abnormal_data.shape[1] * rolling_abnormal_data.shape[2]]), np.reshape(rolling_abnormal_label, [rolling_abnormal_label.shape[0], rolling_abnormal_label.shape[1] * rolling_abnormal_label.shape[2]])
    else:
        if train_data is not None:
            rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = train_data, abnormal_data, abnormal_label
        else:
            rolling_abnormal_data, rolling_abnormal_label = abnormal_data, abnormal_label

    config.x_dim = rolling_abnormal_data.shape[1]

    model = DAGMM(file_name=file_name, config=config)
    model = model.to(device)
    
    if train_data is not None and config.robustness == False:
        dagmm_output = model.fit(train_input=rolling_train_data, train_label=rolling_train_data,
                                 test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                                 abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                                 original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        dagmm_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_data,
                                 test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                                 abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                                 original_x_dim=original_x_dim)
    # %%
    min_max_scaler = preprocessing.StandardScaler()
    if config.preprocessing:
        if config.use_overlapping:
            dec_mean_unroll = unroll_window_3D(np.reshape(dagmm_output.dec_means.detach().cpu().numpy(), (-1, config.rolling_size, original_x_dim)))[::-1]
            energy_unroll = dagmm_output.energy.detach().cpu().numpy()[: dec_mean_unroll.shape[0]]
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
            abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
        else:
            dec_mean_unroll = np.reshape(dagmm_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
            energy_unroll = dagmm_output.energy.detach().cpu().numpy()[: dec_mean_unroll.shape[0]]
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
            abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
    else:
        dec_mean_unroll = dagmm_output.dec_means.detach().cpu().numpy()
        energy_unroll = dagmm_output.energy.detach().cpu().numpy()
        x_original_unroll = abnormal_data
        abnormal_segment = abnormal_label

    if config.save_output:
        if not os.path.exists('./save_outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./save_outputs/NPY/{}/'.format(config.dataset))
        np.save('./save_outputs/NPY/{}/Dec_DAGMM_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), dec_mean_unroll)
        np.save('./save_outputs/NPY/{}/Energy_DAGMM_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), energy_unroll)

    scalerInput = preprocessing.StandardScaler()
    x_original_unroll = scalerInput.fit_transform(x_original_unroll)

    error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2
    final_zscore = zscore(energy_unroll)
    np_decision = create_label_based_on_zscore(final_zscore, 3, True)

    abnormal_segment = abnormal_segment[-np_decision.shape[0]:]
    # TODO metrics computation.

    # %%
    if config.save_figure:
        if not os.path.exists('./save_figures/{}/'.format(config.dataset)):
            os.makedirs('./save_figures/{}/'.format(config.dataset))
        if original_x_dim == 1:
            plt.figure(figsize=(9, 3))
            plt.plot(x_original_unroll, color='blue', lw=1.5)
            plt.title('Original Data')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./save_figures/{}/Ori_DAGMM_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()

            # Plot decoder output
            plt.figure(figsize=(9, 3))
            plt.plot(dec_mean_unroll, color='blue', lw=1.5)
            plt.title('Decoding Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./save_figures/{}/Dec_DAGMM_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()

            plt.figure(figsize=(9, 3))
            plt.plot(energy_unroll, color='blue', lw=1.5)
            plt.title('Energy Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./save_figures/{}/Energy_DAGMM_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()

            t = np.arange(0, abnormal_data.shape[0])
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
            plt.plot(np.squeeze(abnormal_data[: dec_mean_unroll.shape[0]]), alpha=0.7)
            plt.scatter(t[: np_decision.shape[0]], x_original_unroll[: np_decision.shape[0]], s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./save_figures/{}/VisInp_DAGMM_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
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
            plt.savefig('./save_figures/{}/VisOut_DAGMM_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        try:
            pos_label = -1

            if config.adjusted_points:
                label_groups = np.argwhere(np.diff(abnormal_segment.squeeze()))
                if len(label_groups)%2 != 0:
                    label_groups = np.append(label_groups,len(abnormal_segment))

                anomaly_groups = label_groups.squeeze().reshape(-1, 2)

                predicted = pd.DataFrame(np_decision, columns=['Score'])

                for segment in anomaly_groups:
                    predicted_segment = predicted.iloc[segment[0]+1:segment[1]+1]
                    try:
                        if predicted_segment['Score'].value_counts()[1] != segment[1] - segment[0]:
                            predicted.iloc[segment[0]+1:segment[1]+1] = -1
                    except:
                        pass

                np_decision = predicted.to_numpy().squeeze()

            cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
            pre, re, thresholds = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment), probas_pred=np.nan_to_num(energy_unroll), pos_label=pos_label)
            pr_auc = auc(re, pre)
            fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), y_score=np.nan_to_num(energy_unroll), pos_label=pos_label)
            roc_auc = auc(fpr, tpr)       
            cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)  

            if config.use_best_f1:
                metrics = np.array([['Thresholds','Precision','Recall','F1']])

                for threshold in thresholds:
                    y_predicted = []
                    score = error
                    for row in score:
                        if row > threshold:
                            y_predicted.append(-1)
                        else:
                            y_predicted.append(1)   

                    predicted = pd.DataFrame(y_predicted, columns=['Score'])

                    if config.adjusted_points:
                        label_groups = np.argwhere(np.diff(abnormal_segment.squeeze()))
                        if (len(label_groups)%2 != 0):
                            label_groups = np.append(label_groups,len(abnormal_segment))

                        anomaly_groups = label_groups.squeeze().reshape(-1, 2)

                        for segment in anomaly_groups:
                            predicted_segment = predicted.iloc[segment[0]+1:segment[1]+1]
                            try:
                                if (predicted_segment['Score'].value_counts()[1] != segment[1] - segment[0]):
                                    predicted.iloc[segment[0]+1:segment[1]+1] = -1
                            except:
                                pass

                    np_decision = predicted.to_numpy().squeeze()[-np_decision.shape[0]:]
                    
                    precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                    recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                    fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
                    metrics = np.concatenate((metrics, [[threshold, precision, recall, fbeta]]), axis=0)

                topF1 = pd.DataFrame(data=metrics[1:,0:],columns=metrics[0,0:]).apply(pd.to_numeric).nlargest(1, ['F1'])

                precision = topF1['Precision'].values[0]
                recall = topF1['Recall'].values[0]
                fbeta = topF1['F1'].values[0]
            else:
                precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
            
            settings = config.to_string()
            insert_sql = """INSERT or REPLACE into model (model_name, pid, settings, dataset, file_name, TN, FP, FN, 
            TP, precision, recall, fbeta, pr_auc, roc_auc, cks, best_TN, best_FP, best_FN, best_TP, best_precision, 
            best_recall, best_fbeta, best_pr_auc, best_roc_auc, best_cks) VALUES('{}', '{}', '{}', '{}', '{}', '{}', 
            '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', 
            '{}', '{}')""".format(
                'DAGMM', config.pid, settings, config.dataset, Path(file_name).stem, cm[0][0], cm[0][1], cm[1][0],
                cm[1][1], precision, recall, fbeta, pr_auc, roc_auc, cks, dagmm_output.best_TN, dagmm_output.best_FP,
                dagmm_output.best_FN, dagmm_output.best_TP, dagmm_output.best_precision, dagmm_output.best_recall,
                dagmm_output.best_fbeta, dagmm_output.best_pr_auc, dagmm_output.best_roc_auc, dagmm_output.best_cks)
            cursor_obj.execute(insert_sql)
            conn.commit()
            metrics_result = MetricsResult(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1], precision=precision,
                                           recall=recall, fbeta=fbeta, pr_auc=pr_auc, roc_auc=roc_auc, cks=cks,
                                           best_TN=dagmm_output.best_TN, best_FP=dagmm_output.best_FP,
                                           best_FN=dagmm_output.best_FN, best_TP=dagmm_output.best_TP,
                                           best_precision=dagmm_output.best_precision, best_recall=dagmm_output.best_recall,
                                           best_fbeta=dagmm_output.best_fbeta, best_pr_auc=dagmm_output.best_pr_auc,
                                           best_roc_auc=dagmm_output.best_roc_auc, best_cks=dagmm_output.best_cks)
            return metrics_result
        except Exception as e:
            print(e)
            pass


if __name__ == '__main__':
    conn = sqlite3.connect('./experiments.db')
    cursor_obj = conn.cursor()

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--n_gmm', type=int, default=3)
    parser.add_argument('--preprocessing', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    parser.add_argument('--rolling_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--early_stopping', type=str2bool, default=False)
    parser.add_argument('--use_clip_norm', type=str2bool, default=True)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--lambda_energy', type=float, default=1)
    parser.add_argument('--lambda_cov', type=float, default=1)
    parser.add_argument('--display_epoch', type=int, default=5)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=True)
    parser.add_argument('--save_model', type=str2bool, default=True)  # save model
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--continue_training', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--use_last_point', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--adjusted_points', type=str2bool, default=True)
    parser.add_argument('--use_best_f1', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    if args.load_config:
        config = DAGMMConfig(dataset=None, x_dim=None, h_dim=None, z_dim=None, n_gmm=None, preprocessing=None,
                             use_overlapping=None, rolling_size=None, epochs=None, milestone_epochs=None, lr=None,
                             gamma=None, batch_size=None, weight_decay=None, early_stopping=None, use_clip_norm=None,
                             gradient_clip_norm=None, lambda_energy=None, lambda_cov=None, display_epoch=None,
                             save_output=None, save_figure=None, save_model=None, load_model=None, continue_training=None,
                             dropout=None, use_spot=None, use_last_point=None,save_config=None,load_config=None, 
                             server_run=None,robustness=None, adjusted_points=None, use_best_f1=None, pid=None)
        try:
            config.import_config('./save_configs/{}/Config_DAGMM_pid={}.json'.format(config.dataset, config.pid))
        except:
            print('There is no config.')
    else:
        config = DAGMMConfig(dataset=args.dataset, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
                             n_gmm=args.n_gmm, preprocessing=args.preprocessing, use_overlapping=args.use_overlapping,
                             rolling_size=args.rolling_size, epochs=args.epochs, milestone_epochs=args.milestone_epochs,
                             lr=args.lr, gamma=args.gamma, batch_size=args.batch_size, weight_decay=args.weight_decay,
                             early_stopping=args.early_stopping, use_clip_norm=args.use_clip_norm,
                             gradient_clip_norm=args.gradient_clip_norm, lambda_energy=args.lambda_energy,
                             lambda_cov=args.lambda_cov, display_epoch=args.display_epoch, save_output=args.save_output,
                             save_figure=args.save_figure, save_model=args.save_model, load_model=args.load_model,
                             continue_training=args.continue_training, dropout=args.dropout, use_spot=args.use_spot,
                             use_last_point=args.use_last_point, save_config=args.save_config,
                             load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                             adjusted_points=args.adjusted_points, use_best_f1=args.use_best_f1, pid=args.pid)
    if args.save_config:
        if not os.path.exists('./save_configs/{}/'.format(config.dataset)):
            os.makedirs('./save_configs/{}/'.format(config.dataset))
        config.export_config('./save_configs/{}/Config_DAGMM_pid={}.json'.format(config.dataset, config.pid))
    # %%
    device = torch.device(get_free_device())

    train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                           train_logger_name='dagmm_train_logger',
                                                           file_logger_name='dagmm_file_logger',
                                                           meta_logger_name='dagmm_meta_logger',
                                                           model_name='DAGMM',
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
                meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
                meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
                meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
                meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
                meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
                meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
                meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
                meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
            except Exception as e:
                send_email_notification(
                    subject='baseline_dagmm application CRASHED! dataset error: {}, file error: {}, '
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
            meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
            meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
            meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
            meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
            meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
            meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
            meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
            meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
            meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
            meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
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
                meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
                meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
                meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
                meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
                meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
                meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
                meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
                meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
            except Exception as e:
                send_email_notification(
                    subject='baseline_dagmm application CRASHED! dataset error: {}, file error: {}, '
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
            meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
            meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
            meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
            meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
            meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
            meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
            meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
            meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
            meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
            meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
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
                meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
                meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
                meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
                meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
                meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
                meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
                meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
                meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
            except Exception as e:
                send_email_notification(
                    subject='baseline_dagmm application CRASHED! dataset error: {}, file error: {}, '
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
            meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
            meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
            meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
            meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
            meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
            meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
            meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
            meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
            meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
            meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
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
                        except Exception as e:
                            send_email_notification(
                                subject='baseline_dagmm application CRASHED! dataset error: {}, file error: {}, '
                                        'pid: {}'.format(args.dataset, file_name, args.pid),
                                message=str(traceback.format_exc()) + str(e))
                            continue
                    else:
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
                        except Exception as e:
                            file_logger.info(e)
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
                # logger.shutdown()
                meta_logger.info('Finish')
                # meta_logger.shutdown()

    if args.server_run:
        send_email_notification(
            subject='baseline_dagmm application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid),
            message='baseline_dagmm application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid))