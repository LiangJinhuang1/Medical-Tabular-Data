from ast import List
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar
import numpy as np
from .util import PartKey



def load_data(path: str | Path)->dict[PartKey, np.ndarray]:
    path = Path(path).resolve()
    return{
            part: np.load(path / f'X_{part}.npy', allow_pickle=True)
            for part in ['train','val','test']
            if path.jsonpath(f'X_{part}.npy').exists
    }
