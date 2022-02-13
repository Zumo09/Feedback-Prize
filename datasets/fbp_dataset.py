import os
from tqdm import tqdm
from functools import reduce
from typing import Dict, List, Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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

        tag_boxes = torch.Tensor([doc_tags["box_center"], doc_tags["box_length"]]).T

        target = {"labels": tag_cats, "boxes": tag_boxes}
        info = {"id": doc_name, "length": len_sequence}

        return document, target, info  # type: ignore


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


if __name__ == "__main__":
    doc, tag = load_texts("./input/feedback-prize-2021/", [])
    encoder = OrdinalEncoder().fit(tag["discourse_type"].unique().reshape(-1, 1))
    dataset = FBPDataset(doc, tag, encoder)

    _, targ, _ = dataset[0]
    print(targ["boxes"].size())
