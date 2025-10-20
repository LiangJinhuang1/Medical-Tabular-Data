from ast import arg
from curses import noecho
from dataclasses import dataclass
import enum
import hashlib
import json
from logging import info
from math import isnan
from multiprocessing import Value
from nis import cat
from ntpath import exists
import pickle
from collections.abc import Iterable
from pathlib import Path
import random
from statistics import mean
from turtle import RawTurtle
from typing import Any, Generic, TypeVar, cast
from uu import encode
from winsound import MessageBeep
import numpy as np
import sklearn.preprocessing
import torch
from loguru import logger
from torch import Tensor

from . import env
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
        return Task({
            part: np.load(path/ f'Y_{part}.npy') for part in ['train','val','test']
        },
        task_type,
        score,
        )
    
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


def load_data(path: str | Path)->dict[DataKey,dict[PartKey, np.ndarray]]:
    path = Path(path).resolve()
    return{
            key.lower(): {
            part: np.json.load(path / f'{key}_{part}.npy', allow_pickle=True)
            for part in ['train','val','test']}
            for key in ['X_num','X_bin','X_cat',Y]
            if path.jsonpath(f'{key}_train.npy').exists
    }


T = TypeVar('T', np.ndarray, Tensor)


@dataclass
class Dataset(Generic[T]):
    data : dict[DataKey, dict[PartKey,T]]
    task: Task

    @classmethod
    def from_dir(cls, path: str | Path)->'Dataset[np.ndarray]':
        return Dataset(load_data(path), Task.from_dir(path))
    
    def __post_init__(self)->None:
        """validate the data"""
        data = self.data
        is_numpy = self._is_numpy

        #data type
        for key, allowed_dtype in {
            'x_num': [np.dtype('float32')] if is_numpy else [torch.float32],
            'x_bin': [np.dtype('floar32')] if is_numpy else [torch.float32],
            'x_cat': [] if is_numpy else [torch.int64],
            'y': ([np.dtype('float32'), np.dtype('float64'), np.dtype('int64')]
            if is_numpy else [torch.float32, torch.int64]), 
        }.items():
           if key in data:
            for part, value in data[key].items():
                assert value.dtype in (
                    np.dtype('int32'),
                    np.dtype('int64')
                ) or isinstance(
                    value.dtype, 
                    np.dtypes.StrDtype
                )
            else:
                assert value.dtype in allowed_dtype, (
                    f'The value data[{key}][{part}】 has dtype'
                    f'{value.dtype}, but it must be one of {allowed_dtype}'
                )
        #fix the data types
        if self.task.is_regression:
            for key in data['y']:
                data['y'][key] = (
                    data['y'][key].astype('float32')
                    if self._is_numpy 
                    else data['y'][key].to(torch.float32)
                )
        if 'x_cat' in data and data['x_cat']['train'].dtype == np.dtype('int32'):
            for key in data['x_cat']:
                data['x_cat'][key] = data['x_cat'][key].astype('int64')
        
        #check missing data
        isnan = np.isnan if is_numpy else torch.isnan
        for key in ['x_num','x_bin']:
            if key in data:
                for part,value in data['y'].items():
                    assert not isnan(value).any(), f'data[{key}][{part}] contains NaN values'
        for part, vale in data['y'].items():
            assert not isnan(value).any(), f'data[{key}][{part}] contains NaN values'
        
    def _is_numpy(self)->bool:
        return isinstance(self.data['y']['train'], np.ndarray)

    def __contains__(self, key: DataKey)->bool:
        return key in self.data

    def __getitem__(self, key: DataKey)->dict[PartKey, T]:
        return self.data[key]
    
    def __setitem__(self, key: DataKey, value: dict[PartKey, T])->None:
        self.data[key] = value

    @property
    def n_num_features(self)->int:
        return self.data['x_num']['train'].shape[1] if 'x_num' in self.data else 0
    
    @property
    def n_bin_features(self)->int:
        return self.data['x_bin']['train'].shape[1] if 'x_bin' in self.data else 0
    
    @property
    def n_cat_features(self)->int:
        return self.data['x_cat']['train'].shape[1] if 'x_cat' in self.data else 0

    @property
    def n_features(self)->int:
        return self.n_num_features + self.n_bin_features + self.n_cat_features
    
    def size(self, part:None | PartKey)->int:
        return (
            sum(map(len, self.data['y'].values()))
            if part is None else len(self.data['y'][part])
        )
    

    def parts(self)->Iterable[PartKey]:
        return self.data['y'].keys()

    def compute_cat_cardinalities(self)->list[int]:
        x_cat = self.data.get('x_cat')
        if x_cat is None:
            return []
        unique=np.unique if self._is_numpy else torch.unique
        return ([]
            if x_cat is None 
            else [len(unique(column)) for column in x_cat['train'].T])
    
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


class CatPolicy(enum.Enum):
    ORDINAL = 'ordinal'
    ONE_HOT = 'one_hot'


def transform_cat(
    X_cat:dict[PartKey,np.ndarray], policy: None | str | CatPolicy
)-> dict[PartKey, np.ndarray]:
    if policy is None:
        return X_cat
    
    policy = CatPolicy(policy)

    #ordinal encoding be the first 
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown = 'use_encoded_value',
        unknown_value = unknown_value,
        dtype = 'int64'
    ).fit(X_cat['train'])
    X_cat = {k: encoder.transform(v) for k,v in X_cat.items()}
    max_values = X_cat['train'].max(axis=0)
    for part in ['val','test']:
        part = cast(PartKey, part)
        for column_idx in range(X_cat[part].shape[1]):
            X_cat[part][X_cat[part][:, column_idx]== unknown_value, column_idx] =(
                max_values[column_idx] + 1
            )
    if policy == CatPolicy.ORDINAL:
        return X_cat
    elif policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknow = 'ignore',
            sparse = False,
            dtype =np.float32
        )
        encoder.fit(X_cat['train'])
        return {k: cast(np.ndarray, encoder.transform(v)) for k,v in X_cat.items()}
    else:
        raise ValueError(f'Unknown polict = {policy}')

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
    cat_policy: None | str | CatPolicy = None,
    seed: int = 0,
    cache: bool =False
)-> Dataset[np.ndarray]:
    path = Path(path).resolve()
    if cache:
        args = locals()
        args.pop('cache')
        args.pop('path')
        cache_path = env.get_cache_dir()/(
            f'build_dataset_{path.name}__{hashlib.md5(str(args).encode('utf-8')).hexdigest()}.pickle'
        )
        if cache_path.exists():
            cached_args, cached_value = pickle.loads(cache_path.read_bytes())
            assert cached_args == args, f'Hash collisio for {cache_path}'
            logger.info(f'Using cached dataset: {cache_path.name}')
            return cached_value
    else:
        args = None
        cache_path = None
    
    dataset = Dataset.from_dir(path)
    if 'x_num' in dataset.data:
        dataset['x_num'] = transform_num(dataset['x_num'],num_policy, seed)
    if 'x_cat' in dataset.data:
        dataset['x_cat'] = transform_cat(dataset['x_cat'], cat_policy)

    if cache_path is not None:
        cache_path.write_bytes(pickle.dumps(args, cached_value))
    return dataset