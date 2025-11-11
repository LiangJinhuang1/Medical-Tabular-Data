import torch
import torch.nn as nn
from torch import Tensor

class VAEEncoderMLP(nn.Module):
    def __init__(self, encoder, model, use_mu=True, use_log_var=True,freeze_encoder=True) -> None:
        super().__init__()
        self.encoder = encoder
        self.model = model
        self.use_mu = use_mu
        self.use_log_var = use_log_var
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
    def forward(self, x:Tensor) -> Tensor:
        recon, mu, log_var, z = self.encoder(x, training=False)
        inputs_to_mlp = []
        if self.use_mu:
            inputs_to_mlp.append(mu)
        if self.use_log_var:
            inputs_to_mlp.append(log_var)
        if len(inputs_to_mlp) > 1:
            z_combined = torch.cat(inputs_to_mlp, dim=1)
        else:
            z_combined = inputs_to_mlp[0]
        return self.model(z_combined)

