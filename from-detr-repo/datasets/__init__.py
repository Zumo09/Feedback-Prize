from .fbp_dataset import FBPDataset
from .processing_funcs import PIPELINE

def build_dataset(preprocessing: bool = False) -> FBPDataset:
    preprocess = PIPELINE if preprocessing else None
    return FBPDataset(preprocess=preprocess)