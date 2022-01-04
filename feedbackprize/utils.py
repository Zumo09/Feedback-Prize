import os
import pandas as pd

from typing import Tuple

DATASET_FOLDER = 'dataset'

def load_data_csv() -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATASET_FOLDER, 'train.csv'), 
        dtype={'discourse_id': 'int64', 'discourse_start': int, 'discourse_end': int})

def load_file(file_id: str, folder: str = 'train') -> str:
    path = os.path.join(DATASET_FOLDER, folder, file_id + '.txt')
    with open(path, 'r') as f:
        text = f.read()
    return text

def load_texts(folder: str = 'train') -> pd.Series:
    data_path = os.path.join(DATASET_FOLDER, folder)

    def read(filename):
        with open(os.path.join(data_path, filename), 'r') as f:
            text = f.read()
        return text

    return pd.Series({fname.replace('.txt', ''): read(fname) for fname in os.listdir(data_path)})     

def load_dataset() -> Tuple[pd.Series, pd.DataFrame]:
    return load_texts(), load_data_csv()
