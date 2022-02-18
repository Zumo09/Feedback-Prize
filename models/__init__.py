from typing import Optional
import numpy as np
import torch
from .detr import DETR, PrepareInputs
from .matcher import HungarianMatcher
from .criterion import CriterionDETR
from transformers import LEDModel, LEDTokenizerFast  # type: ignore


def build_models(num_classes: int, class_weights: Optional[np.ndarray], args):
    device = torch.device(args.device)

    model = DETR(
        model=LEDModel.from_pretrained("allenai/led-base-16384"),
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
    )

    tokenizer = PrepareInputs(
        tokenizer=LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
    )

    matcher = HungarianMatcher(
        cost_class=1,
        cost_bbox=args.bbox_loss_coef,
        cost_giou=args.giou_loss_coef,
    )

    weight_dict = {
    "loss_ce": 1,
    "loss_bbox": args.bbox_loss_coef,
    "loss_giou": args.giou_loss_coef,
    "loss_overlap": args.overlap_loss_coef,
    }

    losses = ["labels", "boxes", "cardinality", "overlap"]
    criterion = CriterionDETR(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        gamma=args.focal_loss_gamma,
        class_weights=class_weights,
        effective_num=arg.effective_num
    )
    criterion.to(device)

    return tokenizer, model, criterion
