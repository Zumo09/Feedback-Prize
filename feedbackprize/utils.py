from typing import Dict, List, Tuple
import pandas as pd

def read_data_csv() -> pd.DataFrame:
    return pd.read_csv('dataset/train.csv')

def read_file(file_id: str, folder: str = 'train') -> str:
    with open(f'dataset/{folder}/{file_id}.txt', 'r') as f:
        text = f.read()
    return text

def read_dataset() -> Tuple[pd.Series, pd.DataFrame]:
    data = read_data_csv()
    full_text = pd.Series({id: read_file(id) for id in data['id'].unique()})
    return full_text, data