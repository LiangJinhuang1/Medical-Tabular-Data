
from pandas import DataFrame
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple

class Dataset(TorchDataset):
    'Custom Torch Dataset for tabular data'

    def __init__(self, df: DataFrame, target_col :str):
        if target_col not in df.columns:
            raise ValueError(f'Target column {target_col} not found in data')
        
        df = df.copy()
        #eliminate rows with missing target values
        df = df.dropna(subset=[target_col])
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # to do
        #deal with missing data in X
        X = X.fillna(0)

        self.features = torch.tensor(X.values, dtype=torch.float32)
        self.label= torch.tensor(y.values, dtype = torch.float32)


    def __len__(self)-> int:
        return len(self.label)
    
    def __getitem__(self, idx: int)-> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.label[idx]
        return feature, label





