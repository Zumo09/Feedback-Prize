import os
from typing import Tuple
import pandas as pd
from tqdm import tqdm

from .text import text_prepare

DATASET_FOLDER = "../input/feedback-prize-2021"

def load_data_csv() -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(DATASET_FOLDER, 'train.csv'), 
        dtype={'discourse_id': 'int64', 'discourse_start': int, 'discourse_end': int})
    assert isinstance(df, pd.DataFrame)
    return df

def load_file(file_id: str, folder: str = 'train') -> str:
    path = os.path.join(DATASET_FOLDER, folder, file_id + '.txt')
    with open(path, 'r') as f:
        text = f.read()
    return text

def load_texts(folder: str = 'train', preprocess: bool = False) -> pd.Series:
    data_path = os.path.join(DATASET_FOLDER, folder)

    def read(filename):
        with open(os.path.join(data_path, filename), 'r') as f:
            text = f.read()
        if preprocess:
            return text_prepare(text)
        else:
            return text

    return pd.Series({fname.replace('.txt', ''): read(fname) for fname in tqdm(os.listdir(data_path))})     

def load_dataset(preprocess: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    return load_texts(preprocess=preprocess), load_data_csv()