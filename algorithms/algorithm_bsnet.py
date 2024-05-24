from algorithm import Algorithm
from algorithms.bsnet.bs_net_fc import BSNetFC
import torch
from torch.utils.data import TensorDataset, DataLoader


class AlgorithmBSNet(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def get_selected_indices(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bsnet = BSNetFC(self.splits.train_x.shape[1]).to(device)
        optimizer = torch.optim.Adam(bsnet.parameters(), lr=0.00002)
        X_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_train, X_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0
        for epoch in range(100):
            for batch_idx, (X, y) in enumerate(dataloader):
                if X.shape[0] == 1:
                    continue
                optimizer.zero_grad()
                channel_weights, y_hat = bsnet(X)
                mse_loss = self.criterion(y_hat, y)
                norms_for_all_batches = torch.norm(channel_weights, p=1, dim=1)
                l1_loss = torch.mean(norms_for_all_batches)
                loss = mse_loss + l1_loss * 0.01
                loss.backward()
                optimizer.step()
            print(f"Epoch={epoch} MSE={round(mse_loss.item(), 5)}, L1={round(l1_loss.item(), 5)}, LOSS={round(loss.item(), 5)}")
        mean_weight = torch.mean(channel_weights, dim=0)
        band_indx = (torch.argsort(mean_weight, descending=True)).tolist()
        super()._set_all_indices(band_indx)
        selected_indices = band_indx[: self.target_size]
        return bsnet, selected_indices

    def get_name(self):
        return "bsnet"