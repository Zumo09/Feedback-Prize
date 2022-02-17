import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder

from datasets.processing_funcs import PIPELINE
from .fbp_dataset import FBPDataset, load_texts
from .postprocess import FBPPostProcess
from .cw import get_class_weights


def build_fdb_data(args):
    preprocess = [] if args.no_preprocessing else PIPELINE
    documents, tags = load_texts(args.input_path, preprocess, args.dataset_size)  # type: ignore

    encoder = OrdinalEncoder()
    label_unique = np.array(tags["discourse_type"].unique())  # type: ignore
    encoder.fit(label_unique.reshape(-1, 1))

    train_idx, val_idx = train_test_split(
        documents.index, test_size=args.test_size, random_state=args.seed
    )

    train_dataset = FBPDataset(documents[train_idx], tags, encoder, not args.no_align_target)  # type:ignore
    val_dataset = FBPDataset(documents[val_idx], tags, encoder, not args.no_align_target)  # type:ignore

    num_classes = len(label_unique)
    postprocessor = FBPPostProcess(encoder, tags, num_classes)

    if args.no_class_weight:
        class_weights = None
    else:
        class_weights = get_class_weights(train_idx, tags, encoder, args.num_queries)

    return train_dataset, val_dataset, postprocessor, num_classes, class_weights


def collate_fn(batch):
    return tuple(list(i) for i in zip(*batch))
