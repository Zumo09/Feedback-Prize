from torch.utils.data import Dataset
from typing import List, Callable, Optional, Dict
import pandas as pd
import numpy as np
import os
from torch.utils.data.dataset import T_co
from processing_funcs import *
from functools import reduce


PIPELINE = [
    normalize,
    lower,
    replace_special_characters,
    filter_out_uncommon_symbols,
    strip_text
]


class MyDataset(Dataset):
    def __init__(self, path: str = '../input/feedback-prize-2021/train',
                 preprocess: Optional[List[Callable[[str], str]]] = None,
                 category_map: Dict[str, int] = None):
        self.category_map = category_map
        preprocess = preprocess if preprocess else []
        self.documents = self.load_texts(path, preprocess)
        self.documents = self.documents.sample(frac=1)

        types = {'discourse_id': 'int64', 'discourse_start': int, 'discourse_end': int}
        self.tags = pd.read_csv(os.path.join(path, 'train.csv'), dtype=types)

    def __getitem__(self, index) -> T_co:
        doc_name = self.documents.index[index]
        doc_tags = self.tags[self.tags['id'] == doc_name]
        tag_cats = doc_tags['discourse_type'].map(self.category_map).values

        def map_pred(pred: str):
            p = pred.split()
            return int(p[0]), int(p[-1])

        tag_boxes = doc_tags['predictionstring'].map(map_pred).values
        return self.documents[doc_name], np.stack((tag_cats, tag_boxes), axis=1)

    @staticmethod
    def load_texts(path: str, preprocess: List[Callable[[str], str]]) -> pd.Series:
        documents = {}
        for f_name in os.listdir(path):
            doc_name = f_name.replace('.txt', '')
            with open(f_name, 'r') as f:
                text = reduce(lambda txt, f: f(txt), preprocess, f.read())
                documents[doc_name] = text

        return pd.Series(documents)
