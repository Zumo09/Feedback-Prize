import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder

from datasets.processing_funcs import PIPELINE
from .fbp_dataset import FBPDataset, load_texts
from .postprocess import FBPPostProcess


def build_fdb_data(args):
    preprocess = PIPELINE if args.preprocessing else []
    documents, tags = load_texts(args.input_path, preprocess)  # type: ignore

    encoder = OrdinalEncoder()
    label_unique = np.array(tags["discourse_type"].unique())  # type: ignore
    encoder.fit(label_unique.reshape(-1, 1))

    train_idx, val_idx = train_test_split(
        documents.index, test_size=args.test_size, random_state=args.random_state
    )

    train_dataset = FBPDataset(documents[train_idx], tags, encoder)  # type:ignore
    val_dataset = FBPDataset(documents[val_idx], tags, encoder)  # type:ignore

    postprocessor = FBPPostProcess(encoder, tags)

    return train_dataset, val_dataset, postprocessor

def collate_fn(batch):
    return tuple(list(i) for i in zip(*batch))
