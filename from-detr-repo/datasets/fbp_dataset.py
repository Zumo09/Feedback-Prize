import os
from tqdm import tqdm
from functools import reduce
from typing import Dict, List, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import torch
from torch.utils.data import Dataset


class FBPDataset(Dataset):
    def __init__(
        self,
        # path: str = '../input/feedback-prize-2021/train',
        path: str = "../input/feedback-prize-2021/",
        preprocess: Optional[List[Callable[[str], str]]] = None,
        encoder: Optional[OrdinalEncoder] = None,
    ):

        if encoder is None:
            encoder = OrdinalEncoder()
        preprocess = preprocess if preprocess else []
        self.documents = self.load_texts(path, preprocess)
        self.documents = self.documents.sample(frac=1)

        types = {"discourse_id": "int64", "discourse_start": int, "discourse_end": int}
        self.tags = pd.read_csv(os.path.join(path, "train.csv"), dtype=types)

        self.label_unique = np.array(self.tags["discourse_type"].unique())  # type: ignore
        self.encoder = encoder.fit(self.label_unique.reshape(-1, 1))

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index) -> Tuple[str, int, Dict]:
        doc_name = self.documents.index[index]
        doc_tags = self.tags[self.tags["id"] == doc_name]  # type: ignore
        tag_cats = torch.Tensor(
            self.encoder.transform(
                np.array(doc_tags["discourse_type"]).reshape(-1, 1)
            )
        ).squeeze().long()

        document = self.documents[doc_name]
        len_sequence = len(document.split()) # type: ignore
        tag_boxes = self.map_pred(doc_tags["predictionstring"], len_sequence)  

        return document, len_sequence, {"labels": tag_cats, "boxes": tag_boxes} # type: ignore

    # @staticmethod
    # def map_pred(pred, len_sequence):
    #     tag_boxes = []
    #     for p in pred:
    #         p = p.split()
    #         tag_boxes.append([int(p[0]) / len_sequence, int(p[-1]) / len_sequence])
    #     return torch.Tensor(tag_boxes)

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
        # for f_name in os.listdir(path):
        for f_name in tqdm(os.listdir(path + "train/")):
            doc_name = f_name.replace(".txt", "")
            # with open(f_name, 'r') as f:
            with open(path + "train/" + f_name, "r") as f:
                text = reduce(lambda txt, f: f(txt), preprocess, f.read())
                documents[doc_name] = text

        return pd.Series(documents)
