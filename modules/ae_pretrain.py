import os
from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler, Adam

from utils.data_provider import get_loader
from utils.logger import create_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AECNN(nn.Module):
    def __init__(self, file_name, config):
        super(AECNN, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # data info
        self.x_dim = config.x_dim
        self.embedding_dim = config.embedding_dim

        # preprocessing
        self.preprocessing = config.preprocessing

        # optimization info
        self.ae_epochs = config.ae_epochs
        self.ae_milestone_epochs = config.ae_milestone_epochs
        self.ae_lr = config.ae_lr
        self.ae_gamma = config.ae_gamma
        self.ae_batch_size = config.ae_batch_size
        self.ae_weight_decay = config.ae_weight_decay
        self.ae_early_stopping = config.ae_early_stopping

        self.display_epoch = config.display_epoch

        # dropout
        self.ae_dropout = config.ae_dropout

        # pid
        self.pid = config.pid

        self.ae_save_model = config.ae_save_model
        if self.ae_save_model:
            self.ae_save_model_path = './save_models/{}/AECNN_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.ae_save_model_path = None

        self.ae_load_model = config.ae_load_model
        if self.ae_load_model:
            self.ae_load_model_path = './save_models/{}/AECNN_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.ae_load_model_path = None

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.x_dim, out_channels=self.embedding_dim//2, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Dropout(p=self.ae_dropout),
            nn.Conv1d(in_channels=self.embedding_dim//2, out_channels=self.embedding_dim, kernel_size=3, padding=1),
            nn.Sigmoid()).to(device)
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.embedding_dim//2, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Dropout(p=self.ae_dropout),
            nn.Conv1d(in_channels=self.embedding_dim//2, out_channels=self.x_dim, kernel_size=3, padding=1),
            nn.Sigmoid()).to(device)

    def forward(self, X):
        data_encode = self.encoder(X)
        data_decode = self.decoder(data_encode)
        return data_encode, data_decode

    def fit(self, train_input, train_label, test_input, test_label):
        train_logger, file_logger, meta_logger = create_logger(dataset=self.dataset,
                                                               train_logger_name='aecnn_train_logger',
                                                               file_logger_name='aecnn_file_logger',
                                                               meta_logger_name='aecnn_meta_logger',
                                                               model_name='AECNN',
                                                               pid=self.pid)
        loss = nn.MSELoss()
        opt = Adam(self.parameters(), lr=self.ae_lr, weight_decay=self.ae_weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.ae_milestone_epochs, gamma=self.ae_gamma)
        # get batch data

        if self.preprocessing:
            train_input = np.expand_dims(train_input, axis=0)
        else:
            train_input = np.expand_dims(train_input, axis=0)
            train_label = np.expand_dims(train_label, axis=0)
            test_input = np.expand_dims(test_input, axis=0)
            test_label = np.expand_dims(test_label, axis=0)
            train_data = get_loader(input=np.transpose(train_input, (0, 2, 1)), label=train_label, batch_size=1, from_numpy=True, drop_last=True,
                                    shuffle=False)
            test_data = get_loader(input=np.transpose(test_input, (0, 2, 1)), label=test_label, batch_size=1, from_numpy=True, drop_last=True,
                                   shuffle=False)
        epoch_losses = []
        if self.ae_load_model:
            self.load_state_dict(torch.load(self.load_model_path))
        else:
            self.train()
            for epoch in range(self.ae_epochs):
                train_losses = []
                opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # batch_x = torch.unsqueeze(batch_x, dim=0)
                    # opt.zero_grad()  # Set zero gradient
                    batch_x = batch_x.to(device)
                    batch_encode, batch_decode = self.forward(X=batch_x)
                    batch_loss = loss(batch_decode, batch_x)
                    batch_loss.backward()
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))
                if self.ae_early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
        if self.ae_save_model:
            torch.save(self.state_dict(), self.ae_save_model_path)

        self.eval()
        with torch.no_grad():
            data_encodings = []
            data_decodings = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                batch_x = batch_x.to(device)
                data_encoding, data_decoding = self.forward(X=batch_x)
                data_encodings.append(data_encoding)
                data_decodings.append(data_decoding)
            data_encodings = torch.cat(data_encodings, dim=0)
            data_decodings = torch.cat(data_decodings, dim=0)
            return data_encodings, data_decodings


class AEFC(nn.Module):
    def __init__(self, file_name, config):
        super(AEFC, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # data info
        self.x_dim = config.x_dim
        self.embedding_dim = config.embedding_dim

        # preprocessing
        self.preprocessing = config.preprocessing

        # optimization info
        self.ae_epochs = config.ae_epochs
        self.ae_milestone_epochs = config.ae_milestone_epochs
        self.ae_lr = config.ae_lr
        self.ae_gamma = config.ae_gamma
        self.ae_batch_size = config.ae_batch_size
        self.ae_weight_decay = config.ae_weight_decay
        self.ae_early_stopping = config.ae_early_stopping

        self.display_epoch = config.display_epoch

        # dropout
        self.ae_dropout = config.ae_dropout

        # pid
        self.pid = config.pid

        self.ae_save_model = config.ae_save_model
        if self.ae_save_model:
            if not os.path.exists('./save_model/{}/'.format(self.dataset)):
                os.makedirs('./save_model/{}/'.format(self.dataset))
            self.ae_save_model_path = './save_models/{}/AEFC_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.ae_save_model_path = None

        self.ae_load_model = config.ae_load_model
        if self.ae_load_model:
            self.ae_load_model_path = './save_models/{}/AEFC_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.ae_load_model_path = None

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.x_dim, out_features=self.embedding_dim//2),
            nn.ReLU(),
            nn.Dropout(p=self.ae_dropout),
            nn.Linear(in_features=self.embedding_dim//2, out_features=self.embedding_dim),
            nn.Sigmoid()).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim//2),
            nn.ReLU(),
            nn.Dropout(p=self.ae_dropout),
            nn.Linear(in_features=self.embedding_dim//2, out_features=self.x_dim),
            nn.Sigmoid()).to(device)

    def forward(self, X):
        data_encode = self.encoder(X)
        data_decode = self.decoder(data_encode)
        return data_encode, data_decode

    def fit(self, train_input, train_label, test_input, test_label):
        train_logger, file_logger, meta_logger = create_logger(dataset=self.dataset,
                                                               train_logger_name='aefc_train_logger',
                                                               file_logger_name='aefc_file_logger',
                                                               meta_logger_name='aefc_meta_logger',
                                                               model_name='AEFC',
                                                               pid=self.pid)
        loss = nn.MSELoss()
        opt = Adam(list(self.parameters()), lr=self.ae_lr, weight_decay=self.ae_weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.ae_milestone_epochs, gamma=self.ae_gamma)
        # get batch data
        train_data = get_loader(input=train_input, label=train_label, batch_size=self.ae_batch_size, from_numpy=True, drop_last=True,
                                shuffle=True)
        test_data = get_loader(input=test_input, label=test_label, batch_size=self.ae_batch_size, from_numpy=True, drop_last=True,
                               shuffle=False)
        epoch_losses = []
        if self.ae_load_model:
            self.load_state_dict(torch.load(self.ae_load_model_path))
        else:
            self.train()
            for epoch in range(self.ae_epochs):
                train_losses = []
                opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # batch_x = torch.unsqueeze(batch_x, dim=0)
                    # opt.zero_grad()  # Set zero gradient
                    batch_x = batch_x.to(device)
                    batch_encode, batch_decode = self.forward(X=batch_x)
                    batch_loss = loss(batch_decode, batch_x)
                    batch_loss.backward()
                    opt.step()
                    sched.step()
                    train_losses.append(batch_loss.item())
                epoch_losses.append(mean(train_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {}'.format(epoch, epoch_losses[-1]))
                if self.ae_early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
        if self.ae_save_model:
            torch.save(self.state_dict(), self.ae_save_model_path)

        self.eval()
        with torch.no_grad():
            data_encodings = []
            data_decodings = []
            for i, (batch_x, batch_y) in enumerate(test_data):
                batch_x = batch_x.to(device)
                data_encoding, data_decoding = self.forward(X=batch_x)
                data_encodings.append(data_encoding)
                data_decodings.append(data_decoding)
            data_encodings = torch.cat(data_encodings, dim=0)
            data_decodings = torch.cat(data_decodings, dim=0)
            return data_encodings, data_decodings