import pandas as pd
from pathlib import Path
from loguru import logger



def load_data(path: str | Path)-> pd.DataFrame:
    try:
        df = pd.read_csv(path, delimiter=';', index_col=0)
        return df 
    except FileNotFoundError:
        logger.error(f'File not found at {path}')
    except Exception as e:
        logger.error(f'Error loading data from {e}')

    
