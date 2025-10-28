from torch import Tensor
import torch.nn as nn



class Outputhead(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, feats) -> Tensor:
        return self.fc(feats)

