import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, 
        in_dim:int,
        hidden_size:list[int],
        dropout:float,
        batchnorm:bool,
        activation:str="ReLU"):
        super().__init__()
        layers =[] #list[nn.Module]
        prev = in_dim
        for h in hidden_size:
            layes += [nn.Linear(prev,h)]
            if batchnorm: 
                layers += [nn.BtachNorm1d(h)]
                layers += [nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev,1)] # regression head
        self.net = nn.sequential(*layers)

    def forward(self,x:Tensor)->Tensor:
        return self.net(x)

