import torch
import torch.nn as nn
from torch import Tensor


class EncoderEmbedding(nn.Module):
    def __init__(self, encoder, model) -> None:
        super().__init__()
        self.encoder = encoder
        self.model = model


    def forward(self, x:Tensor) -> Tensor:
        encoder_output = self.encoder(x)
        if isinstance(encoder_output, tuple):
            z = encoder_output[0]  # Extract the latent representation
        else:
            z = encoder_output  # If it's not a tuple, use it directly
        out = self.model(z)
        return out