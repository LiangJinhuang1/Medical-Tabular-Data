import numbers
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
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.25,
    random_state: int = 42
    )->None:
    train_csv_path = Path(train_csv_path)
    val_csv_path = Path(val_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    Y_dtype = np.float32

    try:
        logger.info(f'Reading train data from {train_csv_path}')
        df_train = pd.read_csv(train_csv_path)

        X_train = df_train[NUM_FEATURES]
        Y_train = df_train[TARGET_Y].values


        X_train, X_test, Y_train, Y_test = train_set_split(
            X_train,Y_train,test_size=test_size, random_state=random_state, stratify=None
        )

        logger.info(f'Reading val data from {val_csv_path}')
        df_val = pd.read_csv(val_csv_path)

        X_val = df_val[NUM_FEATURES]
        Y_val = df_val[TARGET_Y].values
    except FileNotFoundError as e:
        logger.error(f'File not founf at : {e.filename}')
        return
    except KeyError as e:
        logger.error(f'Column not found in csv file. Make sure {e} is correct')


    splits = {
        'train' : (X_train, Y_train),
        'val' : (X_val, Y_val),
        'test' : {X_test, Y_test}
    }


    for part, (X_part, Y_part) in splits.items():
        logger.info(f'Saving {part} split (size :{len(X_part)})')
        
        X_data = X_part[NUM_FEATURES].value
        np.save(output_dir/f'X_{part}.npy', X_data.astype(np.float32))
        
        np.save(output_dir/f'Y_{part}.npy', Y_part.astype(Y_dtype))

        with open(output_dir/'info.json','w') as f:
            json.dump(TASK_INFO,f, indent=4)
        
        logger.info(f'Successfully created dataset in {output_dir}')