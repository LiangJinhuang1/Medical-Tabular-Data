from ast import List
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Generic, TypeVar
import numpy as np
import torch
from loguru import logger
from torch import Tensor

from .metrics import calculate_metrics as calculate_metrics_
from .util import PartKey, PredictionType, Score, TaskType
from Medical_Tabular_Data import data

_SCORE_SHOULDBE_MAXIMIZED ={
    Score.RMSE: False,
    Score.MAE: False,
    Score.R2: False,
}

@dataclass(frozen=True)
class Task:
    labels: dict[PartKey,np.ndarray]
    type_: TaskType
    score: Score

    @classmethod
    def from_dir(cls,path: str | Path)->'Task':
        path = Path(path).resolve()
        info=json.loads(path.joinpath('info.json').read_text())
        task_type = TaskType.REGRESSION
        score_dir = info.get('score','rmse')
        score = Score(score_dir)
        
        y_data = {
            part: np.load(path/ f'Y_{part}.npy') for part in ['train','val','test']
            if path.joinpath(f'Y_{part}.npy').exists()
        }

        return Task(y_data, task_type,score)
    
    def __post_init__(self)->None:
        assert self.type_ == TaskType.REGRESSION
        assert isinstance(self.score, Score)
        for key,value in self.labels.items():
            if value.dtype not in (np.float32, np.float64):
                logger.warning(f'Regression labels for {key} are not float')
            self.labels[key] = self.labels[key].astype(np.float32)

    @property
    def is_regression(self)->bool:
        return  True

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

