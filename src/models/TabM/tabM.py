import torch 
from torch import Tensor
import torch.nn as nn
from src.models.TabM.InputAdapter import InputAdapter
from src.models.TabM.BackboneMLP import BackboneMLP
from src.models.TabM.Outputhead import Outputhead



class TabM(nn.Module):
    def __init__(
        self, 
        in_dim,
        out_dim,
        hidden_dims,
        k_heads: int=5,
        adapter_dim=None,
        dropout=0.1):
        super().__init__()
        self.k = k_heads

        #input adapter 
        self.input_adapters= nn.ModuleList([
            InputAdapter(in_dim, adapter_dim)
            for _ in range(k_heads)
        ])

        #shared backbone
        backbone_in = adapter_dim if adapter_dim is not None else in_dim
        self.backbone = BackboneMLP(backbone_in, hidden_dims, dropout=dropout)

        #output heads
        self.output_heads = nn.ModuleList([
            Outputhead(self.backbone.out_dim, out_dim)
            for _ in range(k_heads)
        ])

    def forward(self, x: Tensor) -> Tensor:
        preds = []
        for i in range(self.k):
            adapted = self.input_adapters[i](x)
            feats = self.backbone(adapted)
            out = self.output_heads[i](feats)  # shape: (batch_size, out_dim)
            preds.append(out)
        preds = torch.stack(preds, dim=1)  # shape: (batch_size, k, out_dim)
        return preds

    def predict_mean_and_std(self, x):
        """For eval"""
        with torch.no_grad():
            preds = self.forward(x)
            mean = preds.mean(dim=1)
            std = preds.std(dim=1)
        return mean, std
        

