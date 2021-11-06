import logging
from statistics import mean, median
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from utils.data_provider import get_loader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=False):
        super(FullyConnectedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        return x


class ConvolutionalLayer1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(ConvolutionalLayer1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1d_1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1d_1(x))
        return x


class EncodeTimeSeriesToSpectralMatrix(nn.Module):
    def __init__(self, in_features, out_features, epochs, lr, batch_size, display_epoch=10):
        super(EncodeTimeSeriesToSpectralMatrix, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=(self.in_features + self.out_features) * 16),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 16, out_features=(self.in_features + self.out_features) * 8),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 8, out_features=(self.in_features + self.out_features) * 4),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 4, out_features=(self.in_features + self.out_features) * 8),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 8, out_features=(self.in_features + self.out_features) * 16),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 16, out_features=self.out_features),
            # nn.Sigmoid(),
        )
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.display_epoch = display_epoch

    def weights_init(self):
        for p in self.encoder.parameters():
            torch.nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = self.encoder(input)
        return output, 0

    def fit(self, X, train_logger):
        # init optimizer
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=100, gamma=0.95)
        loss = nn.MSELoss()

        # get batch data
        train_data = get_loader(X, None, batch_size=self.batch_size)
        epoch_losses = []

        # train model
        for epoch in range(self.epochs):
            self.train()
            train_losses = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                opt.zero_grad()
                recontructed_batch_X, _ = self.forward(batch_X)
                batch_loss = loss(batch_X, recontructed_batch_X)
                batch_loss.backward()
                opt.step()
                sched.step()
                train_losses.append(batch_loss.item())
            epoch_losses.append(mean(train_losses))
            if epoch % self.display_epoch == 0:
                train_logger.info('T encode epoch={}, T loss={}'.format(epoch, epoch_losses[-1]))
            # if epoch > 1:
            #     if -1e-8 < epoch_losses[-1] - epoch_losses[-2] < 1e-8:
            #         train_logger.info('early break')
            #         break
        # test model
        with torch.no_grad():
            reconstructed_X = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                recontructed_batch_X, _ = self.forward(batch_X)
                reconstructed_X.append(recontructed_batch_X)
            reconstructed_X = torch.stack(reconstructed_X, dim=0)
            return torch.squeeze(reconstructed_X, dim=0), mean(epoch_losses)


class DecodeSpectralMatrixToTimeSeries(nn.Module):
    def __init__(self, in_features, out_features, epochs, lr, batch_size, display_epoch=10):
        '''
        Time series will be here with shape [batch, channel, observation]
        :param in_features: in feature
        :param out_features: out feature
        :param display_epoch: display epoch
        '''
        super(DecodeSpectralMatrixToTimeSeries, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=(self.in_features + self.out_features) * 16),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 16, out_features=(self.in_features + self.out_features) * 8),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 8, out_features=(self.in_features + self.out_features) * 4),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 4, out_features=(self.in_features + self.out_features) * 8),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 8, out_features=(self.in_features + self.out_features) * 16),
            nn.PReLU(),
            nn.Linear(in_features=(self.in_features + self.out_features) * 16, out_features=self.out_features),
            # nn.Sigmoid(),
        )
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.display_epoch = display_epoch

    def weights_init(self):
        for p in self.decoder.parameters():
            torch.nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = self.decoder(input)
        return output, 0

    def fit(self, T_L, train_logger):
        T_L = torch.transpose(T_L, dim0=2, dim1=1)
        # init optimizer
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=100, gamma=0.95)
        loss = nn.MSELoss()
        epoch_losses = []

        # train model
        for epoch in range(self.epochs):
            self.train()
            train_losses = []
            opt.zero_grad()
            reconstruc_T_L, _ = self.forward(T_L)
            batch_loss = loss(T_L, reconstruc_T_L)
            batch_loss.backward()
            opt.step()
            sched.step()
            train_losses.append(batch_loss.item())
            epoch_losses.append(mean(train_losses))
            if epoch % self.display_epoch == 0:
                train_logger.info('T decode epoch={}, T loss={}'.format(epoch, epoch_losses[-1]))
            # if epoch > 1:
            #     if -1e-8 < epoch_losses[-1] - epoch_losses[-2] < 1e-8:
            #         train_logger.info('early break')
            #         break
        # test model
        with torch.no_grad():
            self.eval()
            T_L, _ = self.forward(T_L)
            T_L = torch.transpose(T_L, dim0=2, dim1=1)
            return T_L, mean(epoch_losses)

