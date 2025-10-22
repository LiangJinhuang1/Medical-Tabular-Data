from pathlib import Path
import numpy as np
from loguru import logger
from .load_data import load_data
from .preprocessing import transform_num, NumPolicy
from .core import Dataset, Task



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