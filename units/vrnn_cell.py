import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.planar_normalized_flow import PlanarNormalizingFlow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VRNNCell(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, use_PNF=False, PNF_layers=4):
        super(VRNNCell, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()).to(device)

        # feature-extracting transformations
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
            nn.ReLU()).to(device)

        # encoder
        self.phi_enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()).to(device)
        self.enc_mean = nn.Sequential(
            nn.Linear(self.h_dim, z_dim),
            nn.Sigmoid()).to(device)
        self.enc_std = nn.Sequential(
            nn.Linear(self.h_dim, z_dim),
            nn.Softplus()).to(device)

        # phi_prior
        self.phi_prior = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()).to(device)
        # self.prior_mean = nn.Linear(h_dim, z_dim).to(device)
        self.prior_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Sigmoid()).to(device)
        self.prior_std = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.Softplus()).to(device)

        # decoder
        self.phi_dec = nn.Sequential(
            nn.Linear(self.h_dim + self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()).to(device)
        self.dec_std = nn.Sequential(
            nn.Linear(self.h_dim, self.x_dim),
            nn.Softplus()).to(device)
        self.dec_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.x_dim),
            nn.Sigmoid()).to(device)

        self.use_PNF = use_PNF
        if self.use_PNF:
            self.PNF_layers = PNF_layers
            self.PNF = nn.ModuleList([PlanarNormalizingFlow(n_units=self.z_dim) for i in range(self.PNF_layers)])
        else:
            self.PNF_layers = None
            self.PNF = None

        self.rnn_cell = nn.GRUCell(self.h_dim + self.z_dim, self.h_dim).to(device)

    def forward(self, x, h):
        '''
        :param input_sequence: [batch_size, features]
        :return: h, z
        '''

        # extract
        phi_x = self.phi_x(x)

        # phi_prior
        phi_z_prior = self.phi_prior(h)
        z_prior_mean = self.prior_mean(phi_z_prior)
        z_prior_std = self.prior_std(phi_z_prior)

        # encoder (inference)
        phi_z_infer = self.phi_enc(torch.cat([phi_x, h], dim=1))
        z_infer_mean = self.enc_mean(phi_z_infer)
        z_infer_std = self.enc_std(phi_z_infer)

        # sampling and reparameterization
        z = self.reparameterized_sample(z_infer_mean, z_infer_std)
        z = self.phi_z(z)  # phi_z

        if self.use_PNF:
            for i in range(self.PNF_layers):
                z, _ = self.PNF[i](x=z, compute_y=True, compute_log_det=False)

        # decoder
        phi_dec = self.phi_dec(torch.cat([z, h], dim=1))
        dec_mean = self.dec_mean(phi_dec)
        dec_std = self.dec_std(phi_dec)

        # recurrence
        h = self.rnn_cell(torch.cat([phi_x, z], dim=1), h)

        # All outputs have the shape [batch_size, h_dim|z_dim]
        return h, phi_x, phi_z_prior, z_prior_mean, z_prior_std, phi_z_infer, z_infer_mean, z_infer_std, phi_dec, dec_mean, dec_std

    def reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(device)
        eps = Variable(eps).to(device)
        return eps.mul(std).add_(mean)

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    # def init_weights(self, stdv):
    #     pass
    #
    # def kld_gauss(self, mean_1, std_1, mean_2, std_2):
    #     """Using std to compute KLD"""
    #
    #     kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
    #                    (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
    #                    std_2.pow(2) - 1)
    #     return 0.5 * torch.sum(kld_element)
    #
    # def nll_bernoulli(self, theta, x):
    #     return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))
    #
    # def nll_gauss(self, mean, std, x):
    #     pass




