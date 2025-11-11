from sympy import tensor
import torch 
import torch.nn as nn
from torch import Tensor, std_mean

class TabVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim = 32,
    hidden_dim = [32,32]) -> None:
        super().__init__()
        
        #encoder
        enc = []
        d = input_dim
        for h in hidden_dim:
            enc += [nn.Linear(d,h), nn.ReLU()]
            d = h
        self.enc = nn.Sequential(*enc)
        self.mu = nn.Linear(d, latent_dim)
        self.log_var = nn.Linear(d, latent_dim)
        # decoder
        dec = []
        d = latent_dim
        for h in reversed(hidden_dim):
            dec += [nn.Linear(d,h), nn.ReLU()]
            d = h
        self.dec = nn.Sequential(*dec)
        self.out = nn.Linear(d, input_dim)

    def encode(self, x:Tensor) -> Tensor:
        x = self.enc(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def reparameterize(self, mu:Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def decode(self, z:Tensor) -> Tensor:
        z = self.dec(z)
        return self.out(z)
    
    def forward(self, x:Tensor, training: bool = True) -> Tensor:
        mu, log_var = self.encode(x)
        if training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        recon = self.decode(z)
        return recon, mu, log_var, z
    
    

    

