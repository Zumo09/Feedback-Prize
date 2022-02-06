import os
from tqdm import tqdm
from functools import reduce
from typing import Dict, List, Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from .processing_funcs import PIPELINE


class FBPDataset(Dataset):
    def __init__(
        self,
        documents: pd.Series,
        tags: pd.DataFrame,
        encoder: OrdinalEncoder,
    ):
        self.documents = documents
        self.tags = tags
        self.encoder = encoder

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index) -> Tuple[str, Dict, Dict]:
        doc_name = self.documents.index[index]
        doc_tags = self.tags[self.tags["id"] == doc_name]  # type: ignore
        tag_cats = (
            torch.Tensor(
                self.encoder.transform(
                    np.array(doc_tags["discourse_type"]).reshape(-1, 1)
                )
            )
            .squeeze()
            .long()
        )

        document = self.documents[doc_name]
        len_sequence = len(document.split())  # type: ignore
        tag_boxes = self.map_pred(doc_tags["predictionstring"], len_sequence)

        target = {"labels": tag_cats, "boxes": tag_boxes}
        info = {"id": doc_name, "origin_len": len_sequence}

        return document, target, info # type: ignore

    @staticmethod
    def map_pred(pred, len_sequence):
        tag_boxes = []
        for p in pred:
            p = p.split()
            p = [int(n) for n in p]
            p = torch.Tensor(p)
            tag_boxes.append([torch.mean(p) / len_sequence, p.size()[0] / len_sequence])

        return torch.Tensor(tag_boxes)


class FBPEvaluator:
    def __init__(self, encoder: OrdinalEncoder) -> None:
        self.encoder = encoder
    
    def __call__(self, predictions, targets):
        """
        Evaluate the predictions based on the metrics of Kaggle
        """
        return 0


def load_texts(
    path: str, preprocess: List[Callable[[str], str]]
) -> Tuple[pd.Series, pd.DataFrame]:
    documents = {}
    # for f_name in os.listdir(path):
    for f_name in tqdm(os.listdir(path + "train/")):
        doc_name = f_name.replace(".txt", "")
        # with open(f_name, 'r') as f:
        with open(path + "train/" + f_name, "r") as f:
            text = reduce(lambda txt, f: f(txt), preprocess, f.read())
            documents[doc_name] = text

    types = {"discourse_id": "int64", "discourse_start": int, "discourse_end": int}
    tags = pd.read_csv(os.path.join(path, "train.csv"), dtype=types)

    return pd.Series(documents), tags  # type: ignore


def build_datasets_evaluator(preprocessing: bool, test_size: float, random_state: int) -> Tuple[FBPDataset, FBPDataset, FBPEvaluator]:
    preprocess = PIPELINE if preprocessing else []
    documents, tags = load_texts("../input/feedback-prize-2021/", preprocess) # type: ignore

    encoder = OrdinalEncoder()
    label_unique = np.array(tags["discourse_type"].unique())  # type: ignore
    encoder.fit(label_unique.reshape(-1, 1))

    train_idx, val_idx = train_test_split(documents.index, test_size=test_size, random_state=random_state)

    train_dataset = FBPDataset(documents[train_idx], tags, encoder) # type:ignore
    val_dataset = FBPDataset(documents[val_idx], tags, encoder) # type:ignore

    evaluator = FBPEvaluator(encoder)

    return train_dataset, val_dataset, evaluator
    