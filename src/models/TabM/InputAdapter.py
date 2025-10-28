from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



class InputAdapter(nn.Module):
    def __init__(self, in_dim: int, adapter_dim:int, activation = True) -> None:
        super().__init__()
        if adapter_dim is None:
            adapter_dim = in_dim
        self.fc = nn.Linear(in_dim, adapter_dim)
        self.activation = activation
        self.adapter_dim = adapter_dim

    def forward(self, x:Tensor)-> Tensor:
        x = self.fc(x)
        if self.activation:
            x = F.relu(x)
        return x


