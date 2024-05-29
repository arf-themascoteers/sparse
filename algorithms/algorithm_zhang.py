from algorithm import Algorithm
import torch
from torch.utils.data import TensorDataset, DataLoader
from algorithms.zhang.zhang_net import ZhangNet
import numpy as np
import math
from data_splits import DataSplits
import train_test_evaluator


class AlgorithmZhang(Algorithm):
    def __init__(self, target_size:int, splits:DataSplits, tag, reporter, verbose):
        super().__init__(target_size, splits, tag, reporter, verbose)
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_selected_indices(self):
        class_size = len(np.unique(self.splits.train_y))
        last_layer_input = 100
        zhangnet = ZhangNet(self.splits.train_x.shape[1], class_size, last_layer_input).to(self.device)
        optimizer = torch.optim.Adam(zhangnet.parameters(), lr=0.001, betas=(0.9,0.999))
        X_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(self.splits.train_y, dtype=torch.int32).to(self.device)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0

        for epoch in range(500):
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                channel_weights, sparse_weights, y_hat = zhangnet(X)
                y = y.type(torch.LongTensor).to(self.device)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.l1_loss(channel_weights)
                lambda_value = self.get_lambda(epoch+1)
                loss = mse_loss + lambda_value*l1_loss
                loss.backward()
                optimizer.step()
            if self.verbose:
                print(f"Epoch={epoch} MSE={round(mse_loss.item(), 5)}, L1={round(l1_loss.item(), 5)}, Lambda={lambda_value}, LOSS={round(loss.item(), 5)}")
            mean_weight, all_bands, selected_bands = self.get_indices(channel_weights)
            t_oa, t_aa, t_k, v_oa, v_aa, v_k  = self.validate(zhangnet, selected_bands)
            self.reporter.report_epoch(epoch, mse_loss.item(), l1_loss.item(), lambda_value, loss.item(),t_oa, t_aa, t_k, v_oa, v_aa, v_k ,selected_bands, mean_weight)
        print("Zhang - selected bands and weights:")
        print("".join([str(i).ljust(10) for i in selected_bands]))
        super()._set_all_indices(all_bands)
        return zhangnet, selected_bands

    def get_indices(self, channel_weights):
        mean_weight = torch.mean(channel_weights, dim=0)
        band_indx = (torch.argsort(mean_weight, descending=True)).tolist()
        return mean_weight, band_indx, band_indx[: self.target_size]

    def l1_loss(self, channel_weights):
        channel_weights = torch.sum(channel_weights, dim=1)
        m = torch.mean(channel_weights)
        return m

    def get_name(self):
        return "zhang"

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch/500)

    def validate(self, model, selected_indices):
        X_test = torch.tensor(self.splits.train_x, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(self.splits.train_y, dtype=torch.int32).to(self.device)
        X_test = X_test[:, selected_indices]
        y_hat = model(X_test)
        t_oa, t_aa, t_k = train_test_evaluator.calculate_metrics(y_test.cpu().numpy(), y_hat.argmax(dim=1).cpu().numpy())

        X_test = torch.tensor(self.splits.validation_x, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(self.splits.validation_y, dtype=torch.int32).to(self.device)
        X_test = X_test[:, selected_indices]
        y_hat = model(X_test)
        v_oa, v_aa, v_k = train_test_evaluator.calculate_metrics(y_test.cpu().numpy(), y_hat.argmax(dim=1).cpu().numpy())

        return t_oa, t_aa, t_k, v_oa, v_aa, v_k

