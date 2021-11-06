import argparse
import os
import sqlite3
import traceback
from pathlib import Path
from statistics import mean
import matplotlib as mpl
from utils.device import get_free_device
from utils.mail import send_email_notification
from utils.outputs import OCOutput
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, \
    cohen_kappa_score, roc_auc_score, confusion_matrix, fbeta_score, roc_curve, auc, precision_recall_curve
from torch.optim import Adam, lr_scheduler
from ae_pretrain import AEFC, AECNN
from utils.config import OCConfig
from utils.data_provider import generate_synthetic_dataset, read_GD_dataset, read_HSS_dataset, read_S5_dataset, \
    read_NAB_dataset, read_2D_dataset, read_SMD_dataset, read_ECG_dataset, rolling_window_2D, cutting_window_2D, \
    get_loader, read_SMAP_dataset, read_MSL_dataset
from utils.logger import create_logger
from utils.metrics import calculate_average_metric, MetricsResult
from utils.utils import str2bool, percentile


class OCRNN(nn.Module):
    def __init__(self, file_name, config):
        super(OCRNN, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        self.preprocessing = config.preprocessing

        # data info
        self.x_dim = config.x_dim
        self.encoding_type = config.encoding_type

        if self.encoding_type != 0:
            self.embedding_dim = config.embedding_dim
        else:
            self.embedding_dim = None
        self.w_dim = config.w_dim
        self.V_dim = config.V_dim
        self.cell_type = config.cell_type

        # sequence info
        self.use_overlapping = config.use_overlapping

        # optimization info
        self.loss_function = config.loss_function
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.milestone_epochs = config.milestone_epochs
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.gamma = config.gamma
        self.display_epoch = config.display_epoch
        self.nu = config.nu
        self.early_stopping = config.early_stopping

        # dropout
        self.dropout = config.dropout

        self.continue_training = config.continue_training

        # pid
        self.pid = config.pid

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_models/{}/'.format(self.dataset)):
                os.makedirs('./save_models/{}/'.format(self.dataset))
            self.save_model_path = './save_models/{}/OCRNN_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = './save_models/{}/OCRNN_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

        self.nu = config.nu

        if self.encoding_type != 0:
            if self.cell_type == 'rnn':
                self.w_input = nn.Parameter(torch.rand(self.embedding_dim, self.w_dim))
                self.b_input = nn.Parameter(torch.rand(self.w_dim))
                self.w_hidden = nn.Parameter(torch.rand(self.w_dim, self.w_dim))
                self.b_hidden = nn.Parameter(torch.rand(self.w_dim))
                # self.w_output = nn.Parameter(torch.rand(self.w_dim, self.x_dim))
                # self.b_output = nn.Parameter(torch.rand(self.x_dim))
                self.w_output = nn.Parameter(torch.rand(self.w_dim, 1))
                self.b_output = nn.Parameter(torch.rand(1))
                # self.V_input = nn.Parameter(torch.rand(self.w_dim, self.w_dim))
                # self.V_hidden = nn.Parameter(torch.rand(self.w_dim, self.w_dim))
                # self.V_output = nn.Parameter(torch.rand(self.w_dim, self.V_dim))
            if self.cell_type == 'gru':
                pass
        else:
            if self.cell_type == 'rnn':
                self.w_input = nn.Parameter(torch.rand(self.x_dim, self.w_dim))
                self.b_input = nn.Parameter(torch.rand(self.w_dim))
                self.w_hidden = nn.Parameter(torch.rand(self.w_dim, self.w_dim))
                self.b_hidden = nn.Parameter(torch.rand(self.w_dim))
                # self.w_output = nn.Parameter(torch.rand(self.w_dim, self.x_dim))
                # self.b_output = nn.Parameter(torch.rand(self.x_dim))
                self.w_output = nn.Parameter(torch.rand(self.w_dim, 1))
                self.b_output = nn.Parameter(torch.rand(1))
                # self.V_input = nn.Parameter(torch.rand(self.w_dim, self.w_dim))
                # self.V_hidden = nn.Parameter(torch.rand(self.w_dim, self.w_dim))
                # self.V_output = nn.Parameter(torch.rand(self.w_dim, self.V_dim))
            if self.cell_type == 'gru':
                pass
        self.hidden_w_t = nn.Parameter(torch.rand(1, self.w_dim))
    
    def forward(self, x):
        # w = torch.sigmoid(self.w(x))
        # V = self.V(w)
        # return w, V
        if self.cell_type == 'rnn':
            # hidden_w_t = Variable(torch.rand(x.shape[0], self.w_dim), requires_grad=True).to(device)
            # hidden_V_t = Variable(torch.rand(x.shape[0], self.w_dim), requires_grad=True).to(device)
            hidden = []
            output = []
            for t in range(x.shape[1]):
                output_w_t = torch.matmul(torch.tanh(
                    torch.matmul(self.hidden_w_t, self.w_hidden) + self.b_hidden
                    + torch.matmul(x[:, t], self.w_input) + self.b_input), self.w_output) + self.b_output
                # output_w_t = torch.tanh(torch.matmul(hidden_w_t, self.w_output))
                # hidden_V_t = torch.tanh(torch.matmul(hidden_V_t, self.V_hidden) + torch.matmul(output_w_t, self.V_input))
                # output_V_t = torch.matmul(hidden_V_t, self.V_output)
                # output_w_t = F.softmax(output_w_t)
                output.append(output_w_t)
        return torch.stack(output, 1)
    
    def ocnn_loss(self, x, rho):
        term1 = 0.5 * torch.norm(self.w_input, 'fro') ** 2 + 0.5 * torch.norm(self.w_hidden, 'fro') ** 2 + 0.5 * torch.norm(self.w_output, 'fro') ** 2
        term2 = 0.5 * torch.norm(self.b_input, 'fro') ** 2 + 0.5 * torch.norm(self.b_hidden, 'fro') ** 2 + 0.5 * torch.norm(self.b_output, 'fro') ** 2
        term3 = 1 / self.nu * torch.mean(torch.relu(rho - self.forward(x)))
        term4 = -rho
        loss = term1 + term2 + term3 + term4
        return loss

    def nndd_loss(self, x, c, R):
        term1 = 0.5 * torch.norm(self.w_input, 'fro') ** 2 + 0.5 * torch.norm(self.w_hidden, 'fro') ** 2 + 0.5 * torch.norm(self.w_output, 'fro') ** 2
        term2 = 0.5 * torch.norm(self.b_input, 'fro') ** 2 + 0.5 * torch.norm(self.b_hidden, 'fro') ** 2 + 0.5 * torch.norm(self.b_output, 'fro') ** 2
        term3 = 1 / self.nu * torch.mean(torch.relu(torch.norm(self.forward(x) - c, 'fro', dim=1) ** 2 - R ** 2))
        term4 = R ** 2
        loss = term1 + term2 + term3 + term4
        return loss

    def simple_nndd_los(self, x, c):
        term1 = 0.5 * torch.norm(self.w_input, 'fro') ** 2 + 0.5 * torch.norm(self.w_hidden, 'fro') ** 2 + 0.5 * torch.norm(self.w_output, 'fro') ** 2
        term2 = 0.5 * torch.norm(self.b_input, 'fro') ** 2 + 0.5 * torch.norm(self.b_hidden, 'fro') ** 2 + 0.5 * torch.norm(self.b_output, 'fro') ** 2
        term3 = torch.mean(torch.norm(self.forward(x) - c, 'fro') ** 2)
        loss = term1 + term2 + term3
        return loss

    def init_center_c(self, train_loader, eps=0.05):
        c = torch.zeros(1, device=device)  # the shape of feature space
        n_samples = 0
        self.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                # get the inputs of the batch
                batch_x = batch_x.to(device)
                phi_batch_x = self.forward(x=batch_x)
                n_samples += phi_batch_x.shape[1]
                c += torch.sum(torch.squeeze(phi_batch_x, dim=0), dim=0)
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def fit(self, train_input, train_label, test_input, test_label):
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)
        # get batch data
        if self.preprocessing:
            pass
        else:
            self.batch_size = 1
            train_data = get_loader(input=np.expand_dims(train_input, axis=0), label=np.expand_dims(train_label, axis=0),
                                    batch_size=self.batch_size, from_numpy=True, drop_last=True, shuffle=False)
            test_data = get_loader(input=np.expand_dims(test_input, axis=0), label=np.expand_dims(test_label, axis=0),
                                   batch_size=self.batch_size, from_numpy=True, drop_last=True, shuffle=False)
        epoch_losses = []

        if self.loss_function == 'nn-dd':
            file_logger.info('Initializing center c...')
            c = self.init_center_c(train_data)
            file_logger.info('Initializing radius R...')
            R = 0.0
        elif self.loss_function == 'nn-dd-simple':
            file_logger.info('Initializing center c...')
            c = self.init_center_c(train_data)
        elif self.loss_function == 'oc-nn':
            file_logger.info('Initializing rho...')
            rho = 0.0

        if self.load_model == True and self.continue_training == False:
            self.load_state_dict(torch.load(self.load_model_path))
        elif self.load_model == True and self.continue_training == True:
            self.load_state_dict(torch.load(self.load_model_path))
            self.train()
            for epoch in range(self.epochs):
                train_losses = []
                opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # batch_x = torch.unsqueeze(batch_x, dim=0)
                    # opt.zero_grad()  # Set zero gradient
                    batch_x = batch_x.to(device)
                    phi_batch_x = self.forward(x=batch_x)

                    if self.loss_function == 'oc-nn':
                        batch_loss = self.ocnn_loss(x=batch_x, rho=rho)
                    elif self.loss_function == 'nn-dd':
                        batch_loss = self.nndd_loss(x=batch_x, c=c, R=R)
                    elif self.loss_function == 'nn-dd-simple':
                        batch_loss = self.nndd_loss(x=batch_x, c=c)

                # backward + optimize only if in training phase
                batch_loss = batch_loss.mean()
                batch_loss.backward()
                opt.step()
                sched.step()
                train_losses.append(batch_loss.item())

                if self.loss_function == 'oc-nn':
                    rho = percentile(phi_batch_x, q=100 * self.nu)
                elif self.loss_function == 'nn-dd':
                    R = percentile(torch.sqrt(torch.norm(phi_batch_x - c, 'fro', dim=1) ** 2), q=100 * (1 - self.nu))

                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))
                if self.early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
        else:
            self.train()
            for epoch in range(self.epochs):
                train_losses = []
                opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # batch_x = torch.unsqueeze(batch_x, dim=0)
                    # opt.zero_grad()  # Set zero gradient
                    batch_x = batch_x.to(device)
                    phi_batch_x = self.forward(x=batch_x)

                    plt.xlabel('$t$')
                    plt.ylabel('$s$')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.margins(0.1)
                    plt.plot(np.squeeze(phi_batch_x.detach().cpu().numpy()), alpha=0.7)
                    plt.show()

                    if self.loss_function == 'oc-nn':
                        batch_loss = self.ocnn_loss(x=batch_x, rho=rho)
                    elif self.loss_function == 'nn-dd':
                        batch_loss = self.nndd_loss(x=batch_x, c=c, R=R)
                    elif self.loss_function == 'nn-dd-simple':
                        batch_loss = self.nndd_loss(x=batch_x, c=c)

                    # backward + optimize only if in training phase
                    batch_loss = batch_loss.mean()
                    batch_loss.backward()
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())

                if self.loss_function == 'oc-nn':
                    rho = percentile(phi_batch_x, q=100 * self.nu)
                elif self.loss_function == 'nn-dd':
                    R = percentile(torch.sqrt(torch.norm(phi_batch_x - c, 'fro', dim=1) ** 2), q=100 * (1 - self.nu))

                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))
                if self.early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break

        if self.save_model:
            torch.save(self.state_dict(), self.save_model_path)

        self.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_data):
                # batch_x = torch.unsqueeze(batch_x, dim=0)
                # opt.zero_grad()  # Set zero gradient
                batch_x = batch_x.to(device)
                phi_batch_x = self.forward(x=batch_x)

                plt.xlabel('$t$')
                plt.ylabel('$s$')
                plt.grid(True)
                plt.tight_layout()
                plt.margins(0.1)
                plt.plot(np.squeeze(phi_batch_x.detach().cpu().numpy()), alpha=0.7)
                plt.show()

                if self.loss_function == 'oc-nn':
                    rho = percentile(phi_batch_x, q=100 * self.nu)
                    # condition = torch.norm(phi_batch_x - rho, 'fro', dim=0) ** 2
                    condition = phi_batch_x - rho
                    decision = torch.where(condition >= 0, torch.ones(condition.shape).to(device),
                                           torch.empty(condition.shape).to(device).fill_(-1))
                    oc_output = OCOutput(y_hat=phi_batch_x, rho=rho, c=None, R=None, threshold=None, decision=decision)
                    return oc_output
                elif self.loss_function == 'nn-dd':
                    R = percentile(torch.sqrt(torch.norm(phi_batch_x - c, 'fro', dim=1) ** 2), q=100 * (1 - self.nu))
                    condition = torch.norm(phi_batch_x - c, 'fro', dim=0) ** 2 - R ** 2
                    decision = torch.where(condition < 0, torch.ones(condition.shape).to(device),
                                           torch.empty(condition.shape).to(device).fill_(-1))
                    oc_output = OCOutput(y_hat=phi_batch_x, rho=None, c=c, R=R, threshold=None, decision=decision)
                    return oc_output
                elif self.loss_function == 'nn-dd-simple':
                    threshold = percentile(torch.sqrt(torch.norm(phi_batch_x - c, 'fro', dim=1) ** 2), q=100 * (1 - self.nu))
                    condition = torch.norm(phi_batch_x - c, 'fro', dim=1) ** 2 - threshold ** 2
                    decision = torch.where(condition < 0, torch.ones(condition.shape).to(device),
                                           torch.empty(condition.shape).to(device).fill_(-1))
                    oc_output = OCOutput(y_hat=phi_batch_x, rho=None, c=c, R=None, threshold=threshold, decision=decision)
                    return oc_output

class OCCNN(nn.Module):
    def __init__(self, file_name, config):
        super(OCCNN, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        self.preprocessing = config.preprocessing

        # data info
        self.x_dim = config.x_dim
        self.encoding_type = config.encoding_type

        if self.encoding_type != 0:
            self.embedding_dim = config.embedding_dim
        else:
            self.embedding_dim = None
        self.w_dim = config.w_dim
        self.V_dim = config.V_dim

        # sequence info
        self.use_overlapping = config.use_overlapping

        # optimization info
        self.loss_function = config.loss_function
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.milestone_epochs = config.milestone_epochs
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.gamma = config.gamma
        self.display_epoch = config.display_epoch
        self.nu = config.nu
        self.early_stopping = config.early_stopping

        # dropout
        self.dropout = config.dropout

        self.continue_training = config.continue_training

        # pid
        self.pid = config.pid

        self.save_model = config.save_model
        if self.save_model:
            if not os.path.exists('./save_models/{}/'.format(self.dataset)):
                os.makedirs('./save_models/{}/'.format(self.dataset))
            self.save_model_path = './save_models/{}/OCCNN_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = './save_models/{}/OCCNN_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

        self.nu = config.nu

        if self.encoding_type != 0:
            self.w = nn.Parameter(torch.rand(self.w_dim, self.embedding_dim, 1))
            # self.b_1 = nn.Parameter(torch.rand(self.w_dim))
            self.V = nn.Parameter(torch.rand(self.V_dim, self.w_dim, 1))
            # self.b_2 = nn.Parameter(torch.rand(self.V_dim))
        else:
            self.w = nn.Parameter(torch.rand(self.w_dim, self.x_dim, 1))
            # self.b_1 = nn.Parameter(torch.rand(self.w_dim))
            self.V = nn.Parameter(torch.rand(self.V_dim, self.w_dim, 1))
            # self.b_2 = nn.Parameter(torch.rand(self.V_dim))

    def forward(self, x):
        return F.conv1d(torch.sigmoid(F.conv1d(x, self.w, bias=None, padding=0)), self.V, bias=None, padding=0)

    def ocnn_loss(self, x, rho):
        term1 = 0.5 * torch.norm(self.w, 'fro') ** 2 #+ 0.5 * torch.sum(self.b_1 ** 2))
        term2 = 0.5 * torch.norm(self.V, 'fro') ** 2 #+ 0.5 * torch.sum(self.b_2 ** 2))
        term3 = 1 / self.nu * torch.mean(torch.relu(rho - self.forward(x)))
        term4 = -rho
        loss = term1 + term2 + term3 + term4
        return loss

    def nndd_loss(self, x, c, R):
        term1 = 0.5 * torch.norm(self.w, 'fro') ** 2  # + 0.5 * torch.sum(self.b_1 ** 2))
        term2 = 0.5 * torch.norm(self.V, 'fro') ** 2  # + 0.5 * torch.sum(self.b_2 ** 2))
        term3 = 1 / self.nu * torch.mean(torch.relu(torch.norm(self.forward(x) - c, 'fro', dim=2) ** 2 - R ** 2))
        term4 = R ** 2
        loss = term1 + term2 + term3 + term4
        return loss

    def simple_nndd_los(self, x, c):
        term1 = 0.5 * torch.norm(self.w, 'fro') ** 2  # + 0.5 * torch.sum(self.b_1 ** 2))
        term2 = 0.5 * torch.norm(self.V, 'fro') ** 2  # + 0.5 * torch.sum(self.b_2 ** 2))
        term3 = torch.mean(torch.norm(self.forward(x) - c, 'fro') ** 2)
        loss = term1 + term2 + term3
        return loss

    def init_center_c(self, train_loader, eps=0.05):
        c = torch.zeros(self.batch_size, self.V_dim, 1, device=device)
        n_samples = 0
        self.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                # get the inputs of the batch
                batch_x = batch_x.to(device)
                phi_batch_x = self.forward(x=batch_x)
                n_samples += phi_batch_x.shape[2]
                c += torch.unsqueeze(torch.sum(phi_batch_x, dim=2), 2)
        c = torch.div(c, n_samples)
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def fit(self, train_input, train_label, test_input, test_label):
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)
        # get batch data
        if self.preprocessing:
            pass
        else:
            self.batch_size = 1
            train_data = get_loader(input=np.expand_dims(np.transpose(train_input), axis=0), label=np.expand_dims(train_label, axis=0),
                                    batch_size=self.batch_size, from_numpy=True, drop_last=True, shuffle=False)
            test_data = get_loader(input=np.expand_dims(np.transpose(test_input), axis=0), label=np.expand_dims(test_label, axis=0),
                                   batch_size=self.batch_size, from_numpy=True, drop_last=True, shuffle=False)
        epoch_losses = []

        if self.loss_function == 'nn-dd':
            file_logger.info('Initializing center c...')
            c = self.init_center_c(train_data)
            file_logger.info('Initializing radius R...')
            R = 0.0
        elif self.loss_function == 'nn-dd-simple':
            file_logger.info('Initializing center c...')
            c = self.init_center_c(train_data)
        elif self.loss_function == 'oc-nn':
            file_logger.info('Initializing rho...')
            rho = 0.0

        if self.load_model == True and self.continue_training == False:
            self.load_state_dict(torch.load(self.load_model_path))
        elif self.load_model == True and self.continue_training == True:
            self.load_state_dict(torch.load(self.load_model_path))
            self.train()
            for epoch in range(self.epochs):
                train_losses = []
                opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # batch_x = torch.unsqueeze(batch_x, dim=0)
                    # opt.zero_grad()  # Set zero gradient
                    batch_x = batch_x.to(device)
                    phi_batch_x = self.forward(x=batch_x)

                    if self.loss_function == 'oc-nn':
                        batch_loss = self.ocnn_loss(x=batch_x, rho=rho)
                    elif self.loss_function == 'nn-dd':
                        batch_loss = self.nndd_loss(x=batch_x, c=c, R=R)
                    elif self.loss_function == 'nn-dd-simple':
                        batch_loss = self.nndd_loss(x=batch_x, c=c)

                    # backward + optimize only if in training phase
                    batch_loss = batch_loss.mean()
                    batch_loss.backward()
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())

                if self.loss_function == 'oc-nn':
                    rho = percentile(phi_batch_x, q=100 * self.nu)
                elif self.loss_function == 'nn-dd':
                    R = percentile(torch.sqrt(torch.norm(phi_batch_x - c, 'fro', dim=1) ** 2), q=100 * (1 - self.nu))

                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))
                if self.early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
        else:
            self.train()
            for epoch in range(self.epochs):
                train_losses = []
                opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # batch_x = torch.unsqueeze(batch_x, dim=0)
                    # opt.zero_grad()  # Set zero gradient
                    batch_x = batch_x.to(device)
                    phi_batch_x = self.forward(x=batch_x)

                    if self.loss_function == 'oc-nn':
                        batch_loss = self.ocnn_loss(x=batch_x, rho=rho)
                    elif self.loss_function == 'nn-dd':
                        batch_loss = self.nndd_loss(x=batch_x, c=c, R=R)
                    elif self.loss_function == 'nn-dd-simple':
                        batch_loss = self.nndd_loss(x=batch_x, c=c)

                    # backward + optimize only if in training phase
                    batch_loss = batch_loss.mean()
                    batch_loss.backward()
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())

                if self.loss_function == 'oc-nn':
                    rho = percentile(phi_batch_x, q=100 * self.nu)
                elif self.loss_function == 'nn-dd':
                    R = percentile(torch.sqrt(torch.norm(phi_batch_x - c, 'fro', dim=1) ** 2), q=100 * (1 - self.nu))

                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))
                if self.early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break

        if self.save_model:
            torch.save(self.state_dict(), self.save_model_path)

        self.eval()
        with torch.no_grad():
            phi_x = self.forward(x=torch.from_numpy(np.expand_dims(np.transpose(test_input), axis=0)).to(device))
            if self.loss_function == 'oc-nn':
                rho = percentile(phi_x, q=100 * self.nu)
                condition = phi_x - rho
                decision = torch.where(condition >= 0, torch.ones(condition.shape).to(device),
                                                       torch.empty(condition.shape).to(device).fill_(-1))
                oc_output = OCOutput(y_hat=phi_x, rho=rho, c=None, R=None, threshold=None, decision=decision)
                return oc_output
            elif self.loss_function == 'nn-dd':
                R = percentile(torch.sqrt(torch.norm(phi_x - c, 'fro', dim=1) ** 2), q=100 * (1 - self.nu))
                condition = torch.norm(phi_x - c, 'fro', dim=1) ** 2 - R ** 2
                decision = torch.where(condition < 0, torch.ones(condition.shape).to(device),
                                                       torch.empty(condition.shape).to(device).fill_(-1))
                oc_output = OCOutput(y_hat=phi_x, rho=None, c=c, R=R, threshold=None, decision=decision)
                return oc_output
            elif self.loss_function == 'nn-dd-simple':
                threshold = percentile(torch.sqrt(torch.norm(phi_x - c, 'fro', dim=1) ** 2), q=100 * (1 - self.nu))
                condition = torch.norm(phi_x - c, 'fro', dim=1) ** 2 - threshold ** 2
                decision = torch.where(condition < 0, torch.ones(condition.shape).to(device),
                                                       torch.empty(condition.shape).to(device).fill_(-1))
                oc_output = OCOutput(y_hat=phi_x, rho=None, c=c, R=None, threshold=threshold, decision=decision)
                return oc_output

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

    if config.preprocessing:
        if config.use_overlapping:
            rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(abnormal_data, config.rolling_size), \
                                                            rolling_window_2D(abnormal_label, config.rolling_size)
        else:
            rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(abnormal_data, config.rolling_size), \
                                                            cutting_window_2D(abnormal_label, config.rolling_size)

        config.x_dim = rolling_abnormal_data.shape[2]

        rolling_abnormal_data, rolling_abnormal_label = np.squeeze(rolling_abnormal_data), np.squeeze(rolling_abnormal_label)
    else:
        rolling_abnormal_data, rolling_abnormal_label = abnormal_data, abnormal_label

        config.x_dim = rolling_abnormal_data.shape[1]

    if config.encoding_type == 0:
        file_logger.info('No encoding phase')

    elif config.encoding_type == 1:
        encoding_model = AECNN(file_name=file_name, config=config)
        encoding_model = encoding_model.to(device)
        data_encoding, data_decoding = encoding_model.fit(train_input=rolling_abnormal_data,
                                                          train_label=rolling_abnormal_label,
                                                          test_input=rolling_abnormal_data,
                                                          test_label=rolling_abnormal_label)
        if config.ae_save_output:
            if not os.path.exists('./save_outputs/NPY/{}/'.format(config.dataset)):
                os.makedirs('./save_outputs/NPY/{}/'.format(config.dataset))
            data_encoding = data_encoding.detach().cpu().numpy()
            np.save('./save_outputs/{}/data_encoding_AECNN_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), data_encoding)

            data_decoding = data_decoding.detach().cpu().numpy()
            np.save('./save_outputs/{}/data_decoding_AECNN_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), data_decoding)
        rolling_abnormal_data, rolling_abnormal_label = np.transpose(np.squeeze(data_encoding, axis=0), (1, 0)), abnormal_label

    elif config.encoding_type == 2:
        encoding_model = AEFC(file_name=file_name, config=config)
        encoding_model = encoding_model.to(device)
        data_encoding, data_decoding = encoding_model.fit(train_input=rolling_abnormal_data,
                                                          train_label=rolling_abnormal_label,
                                                          test_input=rolling_abnormal_data,
                                                          test_label=rolling_abnormal_label)

        if config.save_output:
            if not os.path.exists('./save_outputs/NPY/{}/'.format(config.dataset)):
                os.makedirs('./save_outputs/NPY/{}/'.format(config.dataset))
            data_encoding = data_encoding.detach().cpu().numpy()
            np.save('./save_outputs/{}/data_encoding_AEFC_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), data_encoding)

            data_decoding = data_decoding.detach().cpu().numpy()
            np.save('./save_outputs/{}/data_decoding_AEFC_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), data_decoding)
        rolling_abnormal_data, rolling_abnormal_label = data_encoding, abnormal_label

    if config.model_type == 'cnn':
        model = OCCNN(file_name=file_name, config=config)
        model = model.to(device)
        oc_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_label,
                              test_input=rolling_abnormal_data, test_label=rolling_abnormal_label)

    elif config.model_type == 'rnn':
        model = OCRNN(file_name=file_name, config=config)
        model = model.to(device)
        oc_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_label,
                              test_input=rolling_abnormal_data, test_label=rolling_abnormal_label)

    # %%

    np_y_hat = oc_output.y_hat.detach().cpu().numpy()
    np_decision = torch.squeeze(oc_output.decision).detach().cpu().numpy()

    if config.save_output:
        if not os.path.exists('./save_outputs/NPY/{}/'.format(config.dataset)):
            os.makedirs('./save_outputs/NPY/{}/'.format(config.dataset))
        np.save('./save_outputs/{}/y_hat_TOCC_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), np_y_hat)
        np.save('./save_outputs/{}/decision_TOCC_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), np_decision)

    # TODO metrics computation.

    # %%
    if config.save_figure:
        if not os.path.exists('./save_figures/{}/'.format(config.dataset)):
            os.makedirs('./save_figures/{}/'.format(config.dataset))
        if original_x_dim == 1:
            t = np.arange(0, abnormal_data.shape[0])
            markercolors = ['blue' if i == 1 else 'red' for i in abnormal_label[: abnormal_data.shape[0]]]
            markersize = [4 if i == 1 else 25 for i in abnormal_label[: abnormal_data.shape[0]]]
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
            plt.savefig('./save_figures/{}/VisInp_TOCC_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=350)
            plt.close()

            plt.figure(figsize=(9, 3))
            plt.xlabel('$t$')
            plt.ylabel('$score$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.plot(np.transpose(np.squeeze(np_y_hat, 0)), alpha=0.7)
            # plt.show()
            plt.savefig('./save_figures/{}/Score_TOCC_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
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
            plt.plot(np.squeeze(abnormal_data), alpha=0.7)
            plt.scatter(t, abnormal_data, s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./save_figures/{}/VisOut_TOCC_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=300)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        try:
            pos_label = -1
            cm = confusion_matrix(y_true=abnormal_label[config.rolling_size - 1:], y_pred=np_decision, labels=[1, -1])
            precision = precision_score(y_true=abnormal_label[config.rolling_size - 1:], y_pred=np_decision, pos_label=pos_label)
            recall = recall_score(y_true=abnormal_label[config.rolling_size - 1:], y_pred=np_decision, pos_label=pos_label)
            fbeta = fbeta_score(y_true=abnormal_label[config.rolling_size - 1:], y_pred=np_decision, pos_label=pos_label, beta=0.5)
            fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_label[config.rolling_size - 1:]), y_score=np.nan_to_num(error), pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_label[config.rolling_size - 1:]), probas_pred=np.nan_to_num(error), pos_label=pos_label)
            pr_auc = auc(re, pre)
            cks = cohen_kappa_score(y1=abnormal_label[config.rolling_size - 1:], y2=np_decision)
            settings = config.to_string()
            insert_sql = """INSERT or REPLACE into model (model_name, pid, settings, dataset, file_name, TN, FP, FN, 
            TP, precision, recall, fbeta, pr_auc, roc_auc, cks, best_TN, best_FP, best_FN, best_TP, best_precision, 
            best_recall, best_fbeta, best_pr_auc, best_roc_auc, best_cks) VALUES('{}', '{}', '{}', '{}', '{}', '{}', 
            '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', 
            '{}', '{}')""".format(
                'TOCC', config.pid, settings, config.dataset, Path(file_name).stem, cm[0][0], cm[0][1], cm[1][0],
                cm[1][1], precision, recall, fbeta, pr_auc, roc_auc, cks, oc_output.best_TN, oc_output.best_FP,
                oc_output.best_FN, oc_output.best_TP, oc_output.best_precision, oc_output.best_recall,
                oc_output.best_fbeta, oc_output.best_pr_auc, oc_output.best_roc_auc, oc_output.best_cks)
            cursor_obj.execute(insert_sql)
            conn.commit()
            metrics_result = MetricsResult(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1], precision=precision,
                                           recall=recall, fbeta=fbeta, pr_auc=pr_auc, roc_auc=roc_auc, cks=cks,
                                           best_TN=oc_output.best_TN, best_FP=oc_output.best_FP,
                                           best_FN=oc_output.best_FN, best_TP=oc_output.best_TP,
                                           best_precision=oc_output.best_precision, best_recall=oc_output.best_recall,
                                           best_fbeta=oc_output.best_fbeta, best_pr_auc=oc_output.best_pr_auc,
                                           best_roc_auc=oc_output.best_roc_auc, best_cks=oc_output.best_cks)
            return metrics_result
        except:
            pass

if __name__ == '__main__':
    conn = sqlite3.connect('./experiments.db')
    cursor_obj = conn.cursor()

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='cnn')
    parser.add_argument('--x_dim', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--w_dim', type=int, default=32)
    parser.add_argument('--V_dim', type=int, default=1)
    parser.add_argument('--cell_type', type=str, default='rnn')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--ae_epochs', type=int, default=50)
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--ae_milestone_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--ae_gamma', type=float, default=0.95)
    parser.add_argument('--ae_lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ae_batch_size', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--ae_weight_decay', type=float, default=1e-6)
    parser.add_argument('--early_stopping', type=str2bool, default=True)
    parser.add_argument('--ae_early_stopping', type=str2bool, default=True)
    parser.add_argument('--preprocessing', type=str2bool, default=False)
    parser.add_argument('--encoding_type', type=int, default=0)
    parser.add_argument('--loss_function', type=str, default='oc-nn')
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    parser.add_argument('--rolling_size', type=int, default=256)
    parser.add_argument('--display_epoch', type=int, default=10)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--ae_save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=True)
    parser.add_argument('--save_model', type=str2bool, default=True)  # save model
    parser.add_argument('--ae_save_model', type=str2bool, default=True)  # save model
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--ae_load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--continue_training', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--ae_dropout', type=float, default=0.2)
    parser.add_argument('--nu', type=float, default=0.005)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    if args.load_config:
        config = OCConfig(dataset=None, model_type=None, x_dim=None, embedding_dim=None, w_dim=None, V_dim=None,
                          cell_type=None, epochs=None, ae_epochs=None, lr=None, ae_lr=None, gamma=None, ae_gamma=None,
                          batch_size=None, ae_batch_size=None, milestone_epochs=None, ae_milestone_epochs=None,
                          weight_decay=None, ae_weight_decay=None, preprocessing=None, encoding_type=None,
                          ae_early_stopping=None, early_stopping=None, loss_function=None, use_overlapping=None,
                          rolling_size=None, display_epoch=None, save_output=None, save_figure=None, save_model=None,
                          ae_save_model=None, load_model=None, ae_load_model=None, continue_training=None, dropout=None,
                          ae_dropout=None, nu=None, use_spot=None, save_config=None, load_config=None, server_run=None,
                          robustness=None, pid=None)
        config.export_config('./save_configs/{}/Config_TOCC_pid={}.json'.format(config.dataset, config.pid))
    else:
        config = OCConfig(dataset=args.dataset, model_type=args.model_type, x_dim=args.x_dim,
                          embedding_dim=args.embedding_dim, w_dim=args.w_dim, V_dim=args.V_dim,
                          cell_type=args.cell_type, epochs=args.epochs, ae_epochs=args.ae_epochs, lr=args.lr,
                          ae_lr=args.ae_lr, gamma=args.gamma, ae_gamma=args.ae_gamma, batch_size=args.batch_size,
                          ae_batch_size=args.ae_batch_size, milestone_epochs=args.milestone_epochs,
                          ae_milestone_epochs=args.ae_milestone_epochs, weight_decay=args.weight_decay,
                          ae_weight_decay=args.ae_weight_decay, preprocessing=args.preprocessing,
                          encoding_type=args.encoding_type, ae_early_stopping=args.ae_early_stopping,
                          early_stopping=args.early_stopping, loss_function=args.loss_function,
                          use_overlapping=args.use_overlapping, rolling_size=args.rolling_size,
                          display_epoch=args.display_epoch, save_output=args.save_output, save_figure=args.save_figure,
                          save_model=args.save_model, ae_save_model=args.ae_save_model, load_model=args.load_model,
                          ae_load_model=args.ae_load_model, continue_training=args.continue_training,
                          dropout=args.dropout, ae_dropout=args.ae_dropout, nu=args.nu, use_spot=args.use_spot,
                          save_config=args.save_config, load_config=args.load_config, server_run=None, robustness=None,
                          pid=args.pid)
    if args.save_config:
        if not os.path.exists('./save_configs/{}/'.format(config.dataset)):
            os.makedirs('./save_configs/{}/'.format(config.dataset))
        config.export_config('./save_configs/{}/Config_TOCC_pid={}.json'.format(config.dataset, config.pid))
    # %%
    device = torch.device(get_free_device())

    train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                           train_logger_name='tocc_train_logger',
                                                           file_logger_name='tocc_file_logger',
                                                           meta_logger_name='tocc_meta_logger',
                                                           model_name='TOCC',
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
                    subject='one_class_tnn application CRASHED! dataset error: {}, file error: {}, pid: {}'.format(
                        args.dataset, file_name, args.pid),
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
                    subject='one_class_tnn application CRASHED! dataset error: {}, file error: {}, pid: {}'.format(
                        args.dataset, file_name, args.pid),
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
                    subject='one_class_tnn application CRASHED! dataset error: {}, file error: {}, pid: {}'.format(
                        args.dataset, file_name, args.pid),
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
                                subject='one_class_tnn application CRASHED! dataset error: {}, file error: {}, '
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
                # logger.shutdown()
                meta_logger.info('Finish')
                # meta_logger.shutdown()

    if args.server_run:
        send_email_notification(
            subject='one_class_tnn application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid),
            message='one_class_tnn application FINISHED! dataset: {}, pid: {}'.format(args.dataset, args.pid))
