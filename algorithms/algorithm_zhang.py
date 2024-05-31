from algorithm import Algorithm
import torch
from torch.utils.data import TensorDataset, DataLoader
from algorithms.zhang.zhang_net import ZhangNet
import numpy as np
import math
from data_splits import DataSplits
import train_test_evaluator


class Algorithm_zhang(Algorithm):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.criterion = torch.nn.CrossEntropyLoss()
        class_size = len(np.unique(self.splits.train_y))
        last_layer_input = 100
        self.zhangnet = ZhangNet(self.splits.train_x.shape[1], class_size, last_layer_input).to(self.device)
        self.total_epoch = 500
        self.epoch = -1
        self.X_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.splits.train_y, dtype=torch.int32).to(self.device)
        self.X_val = torch.tensor(self.splits.validation_x, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(self.splits.validation_y, dtype=torch.int32).to(self.device)

    def get_selected_indices(self):
        optimizer = torch.optim.Adam(self.zhangnet.parameters(), lr=0.001, betas=(0.9,0.999))
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0

        for epoch in range(self.total_epoch):
            self.epoch = epoch
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                channel_weights, sparse_weights, y_hat = self.zhangnet(X)
                y = y.type(torch.LongTensor).to(self.device)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.l1_loss(channel_weights)
                lambda_value = self.get_lambda(epoch+1)
                loss = mse_loss + lambda_value*l1_loss
                loss.backward()
                optimizer.step()
                if batch_idx == 0 and self.epoch%10 == 0:
                    self.report_stats(channel_weights, sparse_weights, epoch, mse_loss, l1_loss, lambda_value, loss)


        print("Zhang - selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))

        return self.zhangnet, self.selected_indices

    def report_stats(self, channel_weights, sparse_weights, epoch, mse_loss, l1_loss, lambda_value, loss):
        _, _, y_hat = self.zhangnet(self.X_train)
        yp = torch.argmax(y_hat, dim=1)
        yt = self.y_train.cpu().detach().numpy()
        yh = yp.cpu().detach().numpy()
        t_oa, t_aa, t_k = train_test_evaluator.calculate_metrics(yt, yh)

        _, _, y_hat = self.zhangnet(self.X_val)
        yp = torch.argmax(y_hat, dim=1)
        yt = self.y_val.cpu().detach().numpy()
        yh = yp.cpu().detach().numpy()
        v_oa, v_aa, v_k = train_test_evaluator.calculate_metrics(yt, yh)

        mean_weight = torch.mean(torch.abs(channel_weights), dim=0)
        means_sparse = torch.mean(torch.abs(sparse_weights), dim=0)
        min_cw = torch.min(mean_weight).item()
        min_s = torch.min(means_sparse).item()
        max_cw = torch.max(mean_weight).item()
        max_s = torch.max(means_sparse).item()
        avg_cw = torch.mean(mean_weight).item()
        avg_s = torch.mean(means_sparse).item()

        l0_cw = torch.norm(mean_weight, p=0).item()
        l0_s = torch.norm(means_sparse, p=0).item()

        mean_weight, all_bands, selected_bands = self.get_indices(channel_weights)
        self.set_all_indices(all_bands)
        self.set_selected_indices(selected_bands)
        oa, aa, k = train_test_evaluator.evaluate_split(self.splits, self)
        means_sparse = torch.abs(torch.mean(sparse_weights, dim=0))

        self.reporter.report_epoch(epoch, mse_loss, l1_loss, lambda_value, loss,
                                   t_oa, t_aa, t_k,
                                   v_oa, v_aa, v_k,
                                   oa, aa, k,
                                   min_cw, max_cw, avg_cw,
                                   min_s, max_s, avg_s,
                                   l0_cw, l0_s,
                                   selected_bands, means_sparse)

    def get_indices(self, channel_weights):
        mean_weight = torch.mean(channel_weights, dim=0)
        abs_mean_weight = torch.abs(mean_weight)
        band_indx = (torch.argsort(abs_mean_weight, descending=True)).tolist()
        return mean_weight, band_indx, band_indx[: self.target_size]

    def l1_loss(self, channel_weights):
        # channel_weights = torch.sum(channel_weights, dim=1)
        # m = torch.mean(channel_weights)
        # return m
        return torch.mean(torch.abs(channel_weights))

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch/self.total_epoch)



