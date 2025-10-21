from ast import List
from dataclasses import dataclass
import enum
import json
from logging import info
from math import isnan
from pathlib import Path
import random
from typing import Any, Generic, TypeVar, cast
import numpy as np
import sklearn.preprocessing
import torch
from loguru import logger
from torch import Tensor

from .metrics import calculate_metrics as calculate_metrics_
from .util import DataKey, PartKey, PredictionType, Score, TaskType
from Medical_Tabular_Data import data

_SCORE_SHOULDBE_MAXIMIZED ={
    Score.Accuracy: True,
    Score.CrossEntropy: True,
    Score.ROC_AUC: True,
    Score.RMSE: False,
    Score.MAE: False,
    Score.R2: False,
}

@dataclass(frozen==True)
class Task:
    labels: dict[PartKey,np.ndarray]
    type_: TaskType
    score: Score

    @classmethod
    def from_dir(cls,path: str | Path)->'Task':
        path = Path(path).resolve()
        info=json.loads(path.joinpath('info.json').read_text())
        task_type = TaskType(info['task_type'])
        score = Score(info['score'])
        if score is None:
            score ={
                TaskType.BINCLASS: Score.Accuracy,
                TaskType.MULTCLASS: Score.Accuracy,
                TaskType.REGRESSION: Score.RMSE,
            }[task_type]
        else:
            score =Score(score)
        
        y_data = {
            part: np.load(path/ f'Y_{part}.npy') for part in ['train','val','test']
            if path.joinpath(f'Y_{part}.npy').exists()
        }

        return Task(y_data, task_type,score)
    
    def __post_init__(self)->None:
        assert isinstance(self.type_, TaskType)
        assert isinstance(self.score, Score)
        if self.is_regression():
            assert all(
                value.dtype in (np.dtype('float32'), np.dtype('float64'))
                for value in self.labels.values()
            ),'Regression labels must have dtype=float32'
            for key in self.labels:
                self.labels[key] = self.labels[key].astype(np.float32)

    @property
    def is_regression(self)->bool:
        return  self.type_ == TaskType.REGRESSION

    @property
    def is_binclass(self)->bool:
        return self.type_ == TaskType.BINCLASS

    @property
    def is_multiclass(self)->bool:
        return self.type_ == TaskType.MULTCLASS

    @property
    def is_classification(self)->bool:
        return self.is_binclass or self.is_multiclass
    
    def compute_n_classes(self)->int:
        assert self.is_binclass or self.is_multiclass
        return len(np.unique(self.labels['train']))
    
    def try_compute_n_classes(self)->None | int:
        return None if self.is_regression else self.compute_n_classes()

    def calculate_metrics(self,
                          predictions: dict[PartKey,np.ndarray],
                          prediction_type: str | PredictionType,
                          )-> dict[PartKey,Any]:
        metrics = {
            part : calculate_metrics_(
                self.labels[part], self.type_, prediction_type
            ) for part in predictions
        }
        for part_metrics in metrics.values():
            part_metrics['score'] = (
                1.0 if _SCORE_SHOULDBE_MAXIMIZED[self.score] else -1.0
            )* part_metrics[self.score.value]
        return metrics


def load_data(path: str | Path)->dict[PartKey, np.ndarray]:
    path = Path(path).resolve()
    return{
            part: np.load(path / f'X_{part}.npy', allow_pickle=True)
            for part in ['train','val','test']
            if path.jsonpath(f'X_{part}.npy').exists
    }


T = TypeVar('T', np.ndarray, Tensor)


@dataclass
class Dataset(Generic[T]):
    data : dict[str, dict[PartKey,T]]
    task: Task
  
    def __post_init__(self)->None:
        """validate the data"""
        is_numpy = self._is_numpy

        x_key = 'x'
        x_dtype = self.data[x_key]['train'].dtype
        #data type
        if is_numpy:
            assert x_dtype == np.dtype('float32'), f'X features must be float32, but got {x_dtype}'
        else:
            assert x_dtype == torch.float32, f'X features must be torch.float32, but got {x_dtype}'
        
        #check missing data
        isnan = np.isnan if is_numpy else torch.isnan
        for part,value in self.data[x_key].items():
            assert not isnan(value).any(), f'data["x"][{part}] contains NaN values'
        for part, vale in data['y'].items():
            assert not isnan(value).any(), f'data["y"][{part}] contains NaN values'
        
        if self.task.is_regression:
            y_dtype = self.data['y']['train'].dtype
            if is_numpy:
                assert y_dtype == np.dtype('float32')
            else:
                assert y_dtype == torch.float32

    def _is_numpy(self)->bool:
        return isinstance(self.data['y']['train'], np.ndarray)

    def __getitem__(self, key: str)->dict[PartKey, T]:
        return self.data[key]

    @property
    def n_features(self)->int:
        return self.data['x']['train'].shape[1]
    
    def size(self, part:None | PartKey)->int:
        return (
            sum(map(len, self.data['y'].values()))
            if part is None else len(self.data['y'][part])
        )
    

    def parts(self)->List[PartKey]:
        return list(self.data['y'].keys())


    def to_torch(self, device:None | str | torch.device)->'Dataset[Tensor]':
        return Dataset(
            {
                key:{
                    part : torch.as_tensor(value).to(device)
                    for part, value in self.data[key].items()
                }
                for key in self.data
            },
            self.task
        )


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
    
@dataclass
class RegressionLabelStats:
    mean: float
    std: float

def standardize_labels(
    y: dict[PartKey, np.ndarray]
)->tuple[dict[PartKey, np.ndarray], RegressionLabelStats]:
    assert y['train'].dtype == np.dtype('float32')
    mean = float(y['train'].mean())
    std = float(y['train'].std())
    return {k: (v-mean)/std for k,v in y.items()}, RegressionLabelStats(
        mean = mean, std= std
    )

def build_dataset(
    path: str | Path,
    *,
    num_policy: None | str | NumPolicy = None,
    seed: int = 0,
)-> Dataset[np.ndarray]:
    path = Path(path).resolve()
    logger.info(f'Building dataset from {path}')

    task = Task.from_dir(path)

    x_data = load_data(path)

    if num_policy is not None:
        logger.info(f'Applying numerical policy {num_policy}')
        x_data = transform_num(x_data, num_policy, seed)
    else:
        logger.info('No numerical policy applied')
        x_data = {k:v[:,mask] for k,v in x_data.items()}
        mask = np.array([len(np.unique(x))>1 for x in x_data['train'].T])
        if not mask.all():
            logger.warning(f'Removing {sum(~mask)} columns with constant values')
            x_data = {k:v[:,mask] for k,v in x_data.items()}
        x_data = {k: v.astype(np.float32) for k,v in x_data.items()}
    
    dataset = Dataset(data={'x':x_data, 'y':task.labels}, task=task)
    logger.info(f'Dataset built successfully')
    return dataset