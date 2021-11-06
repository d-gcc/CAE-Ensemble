import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlanarNormalizingFlow(nn.Module):
    def __init__(self, n_units):

        self.n_units = n_units

        self.w = nn.Parameter(torch.rand(1, self.n_units), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)
        self.u = nn.Parameter(torch.rand(1, self.n_units), requires_grad=True)

        # self.register_parameter('u_hat', None)


    def forward(self, x, compute_y, compute_log_det):
        wu = torch.matmul(self.w, self.u)

        # if self.u_hat is None:
        #     self.u_hat = nn.Parameter(torch.randn(input.size()))

        u_hat = self.u + (-1 + nn.functional.softplus(wu) - wu) * self.w / torch.sum(torch.pow(self.w, 2))

        # x_flatten, s1, s2 = flatten_to_ndims(x, 2)  # x.shape == [?, n_units]
        wxb = torch.add(torch.matmul(x, self.w), self.b)
        tanh_wxb = torch.tanh(wxb)

        # compute y = f(x)
        y = None
        if compute_y:
            y = x + u_hat * tanh_wxb  # shape == [?, n_units]
            # y = unflatten_from_ndims(y, s1, s2)

        log_det = None
        if compute_log_det:
            grad = 1 - torch.square(tanh_wxb)  # dtanh(x)/dx = 1 - tanh^2(x)
            phi = grad * self.w  # shape == [?, n_units]
            u_phi = torch.matmul(phi, u_hat)  # shape == [?, 1]
            det_jac = 1 + u_phi  # shape == [?, 1]
            log_det = torch.log(torch.abs(det_jac))  # shape == [?, 1]
            # log_det = unflatten_from_ndims(tf.squeeze(log_det, -1), s1, s2)

        return y, log_det
