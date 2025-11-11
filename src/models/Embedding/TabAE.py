import torch
import torch.nn as nn
from torch import Tensor

class TabAE(nn.Module):
    def __init__(self, input_dim:int, latent_dim =32, hidden_dim = [32, 12],dropout=0.1) -> None:
        super().__init__()
        enc = []
        d = input_dim
        for h in hidden_dim:
            enc += [
                nn.Linear(d,h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            d = h
        self.encoder = nn.Sequential(*enc, nn.Linear(d, latent_dim))
        
       
        dec = []
        d = latent_dim
        for h in reversed(hidden_dim):
            dec += [
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            d = h
        dec += [nn.Linear(d, input_dim)] 
        self.decoder = nn.Sequential(*dec)

    def forward(self, x:Tensor) -> Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon
