class Conv1DAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, epochs, lr, batch_size, dropout=False, display_epoch=10):
        super(Conv1DAutoEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv1d_ae =nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.hidden_channels * 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.hidden_channels * 8, out_channels=self.hidden_channels * 4, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.hidden_channels * 4, out_channels=self.hidden_channels * 2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.Conv1d(in_channels=self.hidden_channels * 2, out_channels=self.hidden_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.hidden_channels, out_channels=self.hidden_channels * 2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.hidden_channels * 2, out_channels=self.hidden_channels * 4, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.hidden_channels * 4, out_channels=self.hidden_channels * 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=self.hidden_channels * 8, out_channels=self.in_channels, kernel_size=5, stride=1, padding=2, bias=True),
            # nn.Sigmoid(),
        )
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.display_epoch = display_epoch

    def forward(self, input):
        output = self.conv1d_ae(input)
        return output, 0

    def weights_init(self):
        for p in self.conv1d_ae.parameters():
            torch.nn.init.xavier_uniform_(p)

    def fit(self, X, Y, train_logger):
        # init optimizer
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=50, gamma=0.95)
        loss = nn.MSELoss()
        # get batch data
        train_data = get_loader(X, Y, batch_size=self.batch_size)
        epoch_losses = []
        # train model
        self.train()
        for epoch in range(self.epochs):
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
                train_logger.info('outter AE epoch = {} , outter AE loss = {}'.format(epoch, epoch_losses[-1]))
            # if epoch > 1:
            #     if -1e-8 < epoch_losses[-1] - epoch_losses[-2] < 1e-8:
            #         train_logger.info('early break')
            #         break
        # test model
        with torch.no_grad():
            self.eval()
            reconstructed_X = []
            for i, (batch_X, batch_Y) in enumerate(train_data):
                recontructed_batch_X, _ = self.forward(batch_X)
                reconstructed_X.append(recontructed_batch_X)
            reconstructed_X = torch.stack(reconstructed_X, dim=0)
            return torch.squeeze(reconstructed_X, dim=0), mean(epoch_losses)