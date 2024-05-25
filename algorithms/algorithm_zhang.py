from algorithm import Algorithm
import torch
from torch.utils.data import TensorDataset, DataLoader
from algorithms.zhang.zhang_net import ZhangNet
import numpy as np
import math


class AlgorithmZhang(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_selected_indices(self):
        class_size = len(np.unique(self.splits.train_y))
        last_layer_input = 100
        if self.splits.get_name() == "ghisaconus":
            last_layer_input = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zhangnet = ZhangNet(self.splits.train_x.shape[1], class_size, last_layer_input).to(device)
        optimizer = torch.optim.Adam(zhangnet.parameters(), lr=0.001, betas=(0.9,0.999))
        X_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(device)
        y_train = torch.tensor(self.splits.train_y, dtype=torch.int32).to(device)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0
        # y = y.type(torch.LongTensor).to(self.device)
        # y_validation = y_validation.type(torch.LongTensor).to(self.device)

        for epoch in range(500):
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                channel_weights, sparse_weights, y_hat = zhangnet(X)
                y = y.type(torch.LongTensor).to(device)
                mse_loss = self.criterion(y_hat, y)
                l1_loss = self.l1_loss(channel_weights)
                lambda_value = self.get_lambda(epoch+1)
                loss = mse_loss + lambda_value*l1_loss
                loss.backward()
                optimizer.step()
            print(f"Epoch={epoch} MSE={round(mse_loss.item(), 5)}, L1={round(l1_loss.item(), 5)}, Lambda={lambda_value}, LOSS={round(loss.item(), 5)}")
        mean_weight = torch.mean(channel_weights, dim=0)
        band_indx = (torch.argsort(mean_weight, descending=True)).tolist()
        print("Zhang - selected bands and weights:")
        print("".join([str(i).ljust(10) for i in band_indx]))
        print("".join([str(round(mean_weight[i].item(),3)).ljust(10) for i in band_indx]))
        super()._set_all_indices(band_indx)
        selected_indices = band_indx[: self.target_size]
        return zhangnet, selected_indices

    def l1_loss(self, channel_weights):
        channel_weights = torch.sum(channel_weights, dim=1)
        m = torch.mean(channel_weights)
        return m

    def get_name(self):
        return "zhang"

    def get_lambda(self, epoch):
        return 0.0001 * math.exp(-epoch/500)