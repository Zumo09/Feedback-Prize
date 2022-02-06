from .fbp_dataset import FBPDataset
from .processing_funcs import PIPELINE

def build_dataset(args) -> FBPDataset:
    preprocess = PIPELINE if args.preprocessing else None
    return FBPDataset(preprocess=preprocess)