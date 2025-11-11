
from pandas import DataFrame
from sympy import ff
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

class Dataset(TorchDataset):
    'Custom Torch Dataset for tabular data'

    def __init__(self, df: DataFrame, target_col :str, apply_normalization: bool = False):
        if target_col not in df.columns:
            raise ValueError(f'Target column {target_col} not found in data')
        
        df = df.copy()
        #eliminate rows with missing values
        missing_threshold = int(200) 
        null_counts = df.isnull().sum()
        features_cols = [col for col in df.columns if col != target_col]
        features_to_drop = null_counts[features_cols][null_counts >= missing_threshold].index
        if len(features_cols) > 0:
            print(f'Dropping {len(features_to_drop)} features with more than {missing_threshold} missing values')
            df = df.drop(columns=features_to_drop)
        else:
            print('No features to drop')
        df = df.dropna()
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Apply normalization to features if requested
        if apply_normalization:
            scaler = StandardScaler()
            X_values = scaler.fit_transform(X.values)
            print(f'Applied StandardScaler normalization to features')
            print(f'Features shape: {X_values.shape}')
            self.scaler = scaler
        else:
            X_values = X.values
            print(f'No normalization applied to features')
            self.scaler = None

        y_values = y.values
        print(f'Target range: [{y.min():.4f}, {y.max():.4f}]')

        self.features = torch.tensor(X_values, dtype=torch.float32)
        self.label= torch.tensor(y_values, dtype = torch.float32)


    def __len__(self)-> int:
        return len(self.label)
    
    def __getitem__(self, idx: int)-> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.label[idx]
        return feature, label