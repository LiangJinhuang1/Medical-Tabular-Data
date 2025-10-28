from torch import Tensor
import torch.nn as nn


class BackboneMLP(nn.Module):
    def __init__(self,in_dim, hidden_dims, dropout=0.1) -> None:
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev,h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)

