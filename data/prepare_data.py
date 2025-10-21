from math import trunc
from operator import truediv
from os import path
from pyexpat.errors import XML_ERROR_INVALID_ARGUMENT
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selction import train_set_split
from loguru import logger
from typing import List

#Configuration
## to do ##
NUM_FEATURES: List[str] = []
TARGET_Y: str = ''


#task metadata
TASK_INFO = {
    'task_type':'regression',
    'score':'rmse',
}


def create_data_dir_from_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42
    )->None:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Reading data from {csv_path}')

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f'File not found at {csv_path}')
        return

    X = df[*NUM_FEATURES]
    Y = df[TARGET_Y].values

    X_train_val, X_test, Y_train_val, Y_test = train_set_split(
        X,Y,test_size=test_size, random_state=random_state, stratify=Y if 
        TASK_INFO['task_type'] != 'regression' else None
    )
    X_train, X_val, Y_train, Y_val = train_set_split(
        X_train_val,Y_train_val,test_size=val_size, random_state=random_state, stratify=Y_train_val if 
        TASK_INFO['task_type'] != 'regression' else None
    )

    splits = {
        'train' : (X_train, Y_train),
        'val' : (X_val, Y_val),
        'test' : {X_test, Y_test}
    }


    for part, (X_part, Y_part) in splits.items():
        logger.info(f'Saving {part} split (size :{len(X_part)})')
        
        if NUM_FEATURES:
            X_data = X_part[NUM_FEATURES].value
            np.save(output_dir/f'X_{part}.npy', X_data.astype(np.float32))
        

        Y_dtype = np.float32 if TASK_INFO['task_type'] == 'regression' else np.int64
        np.save(output_dir/f'Y_{part}.npy', Y_part.astype(Y_dtype))

        with open(output_dir/'info.json','w') as f:
            json.dump(TASK_INFO,f, indent=4)