class RobustPCA(nn.Module):
    def __init__(self, mu=None, lmbda=None, max_iter=100):
        super(RobustPCA, self).__init__()
        if mu:
            self.mu = mu
        else:
            self.mu = None
        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = None
        if max_iter:
            self.max_iter = max_iter

    def frobenius_norm(self, M):
        return torch.norm(M, p='fro')

    def shrink(self, M, tau):
        return torch.sign(M) * torch.max((torch.abs(M) - tau), torch.zeros(M.shape).to(device))

    def svd_threshold(self, M, tau):
        U, S, V = torch.svd(M)
        return torch.mm(U, torch.mm(torch.diag(self.shrink(S, tau)), torch.transpose(V, dim0=1, dim1=0)))

    def forward(self, X, train_logger):
        S = torch.zeros(X.shape).to(device)
        Y = torch.zeros(X.shape).to(device)
        if self.mu is None:
            self.mu = np.prod(X.shape) / (4 * self.frobenius_norm(X))
        self.mu_inv = 1 / self.mu
        if self.lmbda is None:
            self.lmbda = 1 / np.sqrt(np.max(X.shape))
        _tol = 1E-8 * self.frobenius_norm(X)
        epoch = 0
        epoch_losses = np.Inf
        Sk = S
        Yk = Y
        Lk = torch.zeros(X.shape).to(device)

        while (epoch_losses > _tol) and epoch < self.max_iter:
            Lk = self.svd_threshold(X - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(X - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (X - Lk - Sk)
            epoch_losses = self.frobenius_norm(X - Lk - Sk)
            epoch += 1
            if (epoch % 10) == 0 or epoch == 1 or epoch > self.max_iter or epoch_losses <= _tol:
                train_logger.info('RPCA epoch={}, RPCA loss={}'.format(epoch, epoch_losses))
        return Lk, Sk