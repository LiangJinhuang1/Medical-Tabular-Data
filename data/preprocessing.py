import enum
from pathlib import Path
import numpy as np
import sklearn.preprocessing
from loguru import logger

from .util import PartKey



class NumPolicy(enum.Enum):
    STANDARD = 'standard'
    NOISY_QUANTILE = 'noisy_quantile'

def transform_num(
    X_num:dict[PartKey, np.ndarray], policy: None | NumPolicy, seed: None | int
    )-> dict[PartKey, np.ndarray]:
    if policy is not None:
        policy = NumPolicy(policy)
        X_num_train = X_num['train']
        if policy == NumPolicy.STANDARD:
            normalizer = sklearn.preprocessing.StandardScaler()
        elif policy == NumPolicy.NOISY_QUANTILE:
            normalizer = sklearn.preprocessing.QuantileTransformer(
                n_quantiles = max(min(X_num['train'].shape[0]//30, 1000),10),
                output_distribution = 'normal',
                subsample = 1_000_000_000,
                random_state = seed,
            )
            assert seed is None
            X_num_train = X_num_train + np.random.RandomState(seed).normal(
                0, 1e-5, X_num_train.shape
            ).astype(X_num_train.dtype)
        else:
            raise ValueError(f'Unknown policy = {policy}')

        normalizer.fit(X_num_train)
        X_num = {k: normalizer.transform(v) for k,v in X_num.items()}

        # NaNs are replaced with zeros
    X_num = {k: np.nan_to_num(v) for k,v in X_num.items()}

    # remove columns with constant values
    mask = np.array([len(np.unique(x))>1 for x in X_num['train'].T])
    X_num = {k: v[:,mask] for k,v in X_num.items()}
    X_num = {k: v.astype(np.float32) for k,v in X_num.items()}

    return X_num

