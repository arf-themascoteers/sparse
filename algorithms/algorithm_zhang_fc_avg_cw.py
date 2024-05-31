import torch
from data_splits import DataSplits
from algorithms.algorithm_zhang import Algorithm_zhang
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class Algorithm_zhang_fc_cw(Algorithm_zhang):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.zhangnet.classnet = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.zhangnet.bands,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128, self.zhangnet.number_of_classes)
        ).to(self.device)

    def get_selected_indices(self):
        optimizer = torch.optim.Adam(self.zhangnet.parameters(), lr=0.001, betas=(0.9,0.999))
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0

        all_cws = None

        for epoch in range(self.total_epoch):
            self.epoch = epoch
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                channel_weights, sparse_weights, y_hat = self.zhangnet(X)

                mean_weight, all_bands, selected_bands = self.get_indices(channel_weights)
                self.set_all_indices(all_bands)
                self.set_selected_indices(selected_bands)

                if all_cws is None:
                    all_cws = channel_weights
                else:
                    all_cws = torch.cat((all_cws, mean_weight), 0)

                y = y.type(torch.LongTensor).to(self.device)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.l1_loss(channel_weights)
                lambda_value = self.get_lambda(epoch+1)
                loss = mse_loss + lambda_value*l1_loss
                if batch_idx == 0 and self.epoch%10 == 0:
                    self.report_stats(channel_weights, sparse_weights, epoch, mse_loss, l1_loss, lambda_value, loss)
                loss.backward()
                optimizer.step()

            mean_all_cws = torch.mean(all_cws, dim=0)
            all_cws = None
            self.all_indices = (torch.argsort(mean_all_cws, descending=True)).tolist()
            self.selected_indices = self.all_indices[: self.target_size]

        print("Zhang - selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))

        return self.zhangnet, self.selected_indices