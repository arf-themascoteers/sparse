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
        class_size = len(np.unique(self.splits.bs_train_y))
        last_layer_input = 100
        self.zhangnet = ZhangNet(self.splits.bs_train_x.shape[1], class_size, last_layer_input).to(self.device)

    def get_selected_indices(self):
        optimizer = torch.optim.Adam(self.zhangnet.parameters(), lr=0.001, betas=(0.9,0.999))
        X_train = torch.tensor(self.splits.bs_train_x, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(self.splits.bs_train_y, dtype=torch.int32).to(self.device)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0

        for epoch in range(500):
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
                if batch_idx == 0:
                    _,_,y_hat = self.zhangnet(X_train)
                    yp = torch.argmax(y_hat, dim=1)
                    yt = y_train.cpu().detach().numpy()
                    yh = yp.cpu().detach().numpy()
                    t_oa,t_aa,t_k = train_test_evaluator.calculate_metrics(yt, yh)
                    mean_weight, all_bands, selected_bands = self.get_indices(channel_weights)
                    self.set_all_indices(all_bands)
                    self.set_selected_indices(selected_bands)
                    oa, aa, k  = train_test_evaluator.evaluate_split(self.splits, self)
                    self.reporter.report_epoch(epoch, mse_loss.item(), l1_loss.item(), lambda_value, loss.item(),t_oa,t_aa,t_k,oa,aa,k,selected_bands, mean_weight)

            if self.verbose:
                print(f"Epoch={epoch} MSE={round(mse_loss.item(), 5)}, L1={round(l1_loss.item(), 5)}, Lambda={lambda_value}, LOSS={round(loss.item(), 5)}")
                print(f"Min weight={torch.min(mean_weight).item()}, Max weight={torch.max(mean_weight).item()}, L0 norm={torch.norm(mean_weight, p=0).item()}")

        print("Zhang - selected bands and weights:")
        print("".join([str(i).ljust(10) for i in self.selected_indices]))

        return self.zhangnet, self.selected_indices

    def get_indices(self, channel_weights):
        mean_weight = torch.mean(channel_weights, dim=0)
        abs_mean_weight = torch.abs(mean_weight)
        band_indx = (torch.argsort(abs_mean_weight, descending=True)).tolist()
        return mean_weight, band_indx, band_indx[: self.target_size]

    def l1_loss(self, channel_weights):
        channel_weights = torch.sum(channel_weights, dim=1)
        m = torch.mean(channel_weights)
        return m

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch/500)



