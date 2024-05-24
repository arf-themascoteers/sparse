import torch.nn as nn
import torch
from collections import OrderedDict


class ANN(nn.Module):
    def __init__(self, target_size, class_size,structure=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.class_size = class_size

        init_vals = torch.linspace(0.001,0.99, self.target_size+2)
        self.indices = nn.Parameter(torch.tensor([ANN.inverse_sigmoid_torch(init_vals[i + 1]) for i in range(self.target_size)], requires_grad=True).to(self.device))
        if structure == None:
            structure = [128,64]
        self.linear = self.create_structure(structure)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Number of learnable parameters:", num_params)


    def create_structure(self, structure):
        if structure == None:
            structure = []
        structure = [self.target_size] + structure + [self.class_size]
        odict = OrderedDict()

        for index in range(len(structure)-1):
            odict['fc' + str(index+1)] = nn.Linear(structure[index], structure[index+1])
            if index != len(structure)-2:
                odict['lrelu' + str(index+1)] = nn.LeakyReLU()

        return nn.Sequential(odict)

    @staticmethod
    def inverse_sigmoid_torch(x):
        return -torch.log(1.0 / x - 1.0)

    def forward(self, linterp):
        outputs = linterp(self.get_indices())
        y_hat = self.linear(outputs)
        return y_hat

    def get_indices(self):
        return torch.sigmoid(self.indices)

