class AutoEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=False, display_epoch=10,
                 epochs=500, batch_size=128, lr=0.001):
        super(AutoEncoder, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout = dropout
        self.fc_ae = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.hidden_features * 4, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features * 4, out_features=self.hidden_features * 2, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features * 2, out_features=self.hidden_features, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features, out_features=self.hidden_features * 2, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features * 2, out_features=self.hidden_features * 4, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=self.hidden_features * 4, out_features=self.out_features, bias=True),
            # nn.Sigmoid(),
        )

        self.display_epoch = display_epoch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def weights_init(self):
        for p in self.fc_ae.parameters():
            torch.nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = self.fc_ae(input)
        return output, 0

    def fit(self, X, Y, train_logger):
        # init optimizer
        opt = Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=50, gamma=0.95)
        loss = nn.MSELoss()

        # get batch data
        train_data = get_loader(X, Y, batch_size=self.batch_size)
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
                train_logger.info('inner AE epoch = {}, inner AE loss = {}'.format(epoch, epoch_losses[-1]))
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