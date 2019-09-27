import torch
import torch.nn as nn


class No_Net(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        return x