import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SRNN Model
class SRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(SRNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # encoder  x/u to z, input to latent variable, inference model
        self.enc = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # prior transition of zt-1 to zt
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoder from latent variable to output, from z to x
        self.dec = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid())

        self.rnn = nn.GRU(h_dim + x_dim, h_dim, n_layers, bias)
        self.hidden_state_rnn = nn.GRU(x_dim, h_dim, n_layers, bias)

    def forward(self, x, y):
        # generative and inference model
        all_enc_mean, all_enc_std = [], []  # inference
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0  # KL in ELBO
        nll_loss = 0  # -loglikihood in ELBO

        z_t_sampled = []
        z_t = torch.zeros(self.n_layers, x.size(1), self.h_dim)[-1]

        # computing hidden state in list and x_t & y_t in list outside the loop
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim)
        h_list = []
        x_t_list = []
        y_t_list = []

        for t in range(x.size(0)):
            x_t = x[t]
            y_t = y[t]
            _, h = self.hidden_state_rnn(torch.cat([x_t], 1).unsqueeze(0), h)
            y_t_list.append(y_t)
            x_t_list.append(x_t)
            h_list.append(h[-1])

        # reversing hidden state list
        reversed_h = h_list
        reversed_h.reverse()

        # reversing y_t list
        reversed_y_t = y_t_list
        reversed_y_t.reverse()

        # concat reverse h with reverse x_t
        concat_h_t_y_t_list = []

        for t in range(x.size(0)):  # x.size(0) == y.size(0) == 28
            concat_h_t_y_t = torch.cat([reversed_y_t[t], reversed_h[t]], 1).unsqueeze(0)
            concat_h_t_y_t_list.append(concat_h_t_y_t)

            # compute reverse a_t
        a_t = torch.zeros(self.n_layers, x.size(1), self.h_dim)
        reversed_a_t_list = []
        for t in range(x.size(0)):
            _, a_t = self.rnn(concat_h_t_y_t_list[t], a_t)  # RNN new
            reversed_a_t_list.append(a_t[-1])
        reversed_a_t_list.reverse()

        for t in range(x.size(0)):

            # encoder
            enc_t = self.enc(reversed_a_t_list[t])
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # sampling and reparameterization, sampling from infer network
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t_sampled.append(z_t)

            # prior #transition,
            prior_t = self.prior(h_list[t])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # decoder #emission (generativemodel)
            dec_t = self.dec(torch.cat([z_t, h_list[t]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_bernoulli(dec_mean_t, y[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, nll_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std)

    def forecasting(self, x, step, y):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        z_t_sampled = []
        z_t = torch.zeros(self.n_layers, x.size(1), self.h_dim)[-1]

        # computing hidden state in list and x_t & y_t in list outside the loop
        h = torch.zeros(self.n_layers, x.size(1), self.h_dim)
        h_list = []
        x_t_list = []
        y_t_list = []
        for t in range(x.size(0)):
            x_t = x[t]
            y_t = y[t]
            _, h = self.hidden_state_rnn(torch.cat([x_t], 1).unsqueeze(0), h)
            x_t_list.append(x_t)
            y_t_list.append(y_t)
            h_list.append(h[-1])

        # reversing hidden state list
        reversed_h = h_list
        reversed_h.reverse()

        # reversing y_t list
        reversed_y_t = y_t_list
        reversed_y_t.reverse()

        #         #concat reverse h with reverse x_t
        concat_h_t_y_t_list = []
        for t in range(x.size(0)):
            concat_h_t_y_t = torch.cat([reversed_y_t[t], reversed_h[t]], 1).unsqueeze(0)
            concat_h_t_y_t_list.append(concat_h_t_y_t)

        #         #compute reverse a_t
        a_t = torch.zeros(self.n_layers, x.size(1), self.h_dim)
        reversed_a_t_list = []
        for t in range(x.size(0)):
            _, a_t = self.rnn(concat_h_t_y_t_list[t], a_t)  # RNN new
            reversed_a_t_list.append(a_t[-1])
        reversed_a_t_list.reverse()

        for t in range(x.size(0)):
            x_t = x[t]

            # encoder
            enc_t = self.enc(reversed_a_t_list[t])
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h_list[t])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t_sampled.append(z_t)

            # decoder
            dec_t = self.dec(torch.cat([z_t, h_list[t]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        x_predict = []
        for i in range(step):
            # prior
            prior_t = self.prior(h_list[t])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            z_t_sampled.append(z_t)

            # decoder
            dec_t = self.dec(torch.cat([z_t, h_list[i]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            x_t = dec_mean_t
            x_predict.append(dec_mean_t)

        return x_predict, z_t_sampled

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std) + (x - mean).pow(2) / (2 * std.pow(2)))


if __name__ == '__main__':
    seq_len, batch_size, hidden_size, latent_size, input_size = 7, 20, 64, 16, 4
    size = (seq_len, batch_size, input_size)
    X = torch.autograd.Variable(torch.rand(size), requires_grad=True).to(device)
    srnn = SRNN(x_dim=input_size, h_dim=hidden_size, z_dim=latent_size, n_layers=1, bias=False)
    srnn.to(device)
    kld_loss, nll_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std) = srnn(X, X)
    all_enc_mean = torch.stack(all_enc_mean)
    all_dec_mean = torch.stack(all_dec_mean)
    assert all_enc_mean.shape == (7, 20, 16)
    assert all_dec_mean.shape == (7, 20, 4)
