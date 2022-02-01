from torch.utils.data import Dataset
from typing import List, Callable, Optional
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from processing_funcs import *
from functools import reduce
from sklearn.preprocessing import OrdinalEncoder
import torch

PIPELINE = [
    normalize,
    lower,
    replace_special_characters,
    filter_out_uncommon_symbols,
    strip_text
]


class MyDataset(Dataset):
    def __init__(self, 
                 #path: str = '../input/feedback-prize-2021/train',
                 path: str = '../input/feedback-prize-2021/',
                 preprocess: Optional[List[Callable[[str], str]]] = None,
                 #category_map: Dict[str, int] = None
                 encoder: Optional[OrdinalEncoder] = None):
                #  encoder=OrdinalEncoder()):
        
        if encoder is None:
            encoder = OrdinalEncoder()
        #self.category_map = category_map
        preprocess = preprocess if preprocess else []
        self.documents = self.load_texts(path, preprocess)
        self.documents = self.documents.sample(frac=1)

        types = {'discourse_id': 'int64', 'discourse_start': int, 'discourse_end': int}
        self.tags = pd.read_csv(os.path.join(path, 'train.csv'), dtype=types)
        
        label_unique = np.array(self.tags['discourse_type'].unique()) # type: ignore
        self.encoder = encoder.fit(label_unique.reshape(-1, 1))
    
    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        doc_name = self.documents.index[index]
        doc_tags = self.tags[self.tags['id'] == doc_name] # type: ignore
        tag_cats = torch.Tensor(self.encoder.transform(np.array(doc_tags['discourse_type']).reshape(-1, 1)))
        
        document = self.documents[doc_name]
        tag_boxes = self.map_pred(doc_tags['predictionstring'], len(document.split())) # type: ignore

        return document, torch.hstack((tag_cats, tag_boxes)) # type: ignore
        
    @staticmethod
    def map_pred(pred, len_sequence):
        tag_boxes = []
        for p in pred:
            p = p.split()
            p = [int(n) for n in p]
            p = torch.Tensor(p)
            tag_boxes.append([torch.mean(p) / len_sequence, p.size()[0] / len_sequence])

        return torch.Tensor(tag_boxes)

    @staticmethod
    def load_texts(path: str, preprocess: List[Callable[[str], str]]) -> pd.Series:
        documents = {}
        #for f_name in os.listdir(path):
        for f_name in tqdm(os.listdir(path + 'train/')):
            doc_name = f_name.replace('.txt', '')
            #with open(f_name, 'r') as f:
            with open(path + 'train/'+ f_name, 'r') as f:
                text = reduce(lambda txt, f: f(txt), preprocess, f.read())
                documents[doc_name] = text

        return pd.Series(documents)
