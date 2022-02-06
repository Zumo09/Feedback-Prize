from typing import Tuple
from .fbp_dataset import FBPDataset, FBPEvaluator, build_datasets_evaluator


def build_train_val_datasets_evaluator(
    args,
) -> Tuple[FBPDataset, FBPDataset, FBPEvaluator]:
    return build_datasets_evaluator(args.preprocessing, args.test_size, args.seed)
