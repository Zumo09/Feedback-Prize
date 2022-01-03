from typing import List
import pandas as pd

def read_train_csv() -> pd.DataFrame:
    return pd.read_csv('dataset/train.csv')

def read_file(file_id: str, folder: str = 'train') -> List[str]:
    with open(f'dataset/{folder}/{file_id}.txt', 'r') as f:
        lines = f.readlines()
    return lines