from typing import Optional
import numpy as np
import torch
from .detr import DETR, PrepareInputs
from .matcher import HungarianMatcher
from .criterion import CriterionDETR
from transformers import LEDModel, LEDTokenizerFast  # type: ignore


def build_models(num_classes: int, freqs: Optional[np.ndarray], args):
    device = torch.device(args.device)

    model = DETR(
        model=LEDModel.from_pretrained("allenai/led-base-16384"),
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        class_depth=args.class_depth,
        bbox_depth=args.bbox_depth,
        dropout=args.dropout,
        class_biases=np.log(freqs / (1 - freqs)) if args.init_last_biases and freqs is not None else None,
        init_weight=args.init_weight
    )

    tokenizer = PrepareInputs(
        tokenizer=LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
    )

    criterion = make_criterion(num_classes, freqs, args, device)

    return tokenizer, model, criterion

def make_criterion(num_classes, freqs, args, device):
    matcher = HungarianMatcher(
        cost_class=args.ce_loss_coef,
        cost_bbox=args.bbox_loss_coef,
        cost_giou=args.giou_loss_coef,
    )

    weight_dict = {
        "loss_ce": args.ce_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_overlap": args.overlap_loss_coef,
    }

    losses = args.losses
    
    if args.no_class_weight:
        class_weights = None
    elif args.effective_num:
        class_weights = (1-args.beta)/(1-args.beta**(freqs)) #Class weights are the inverse of the freq
    else:
        class_weights = 1 / freqs if freqs is not None else None         
    
    criterion = CriterionDETR(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        gamma=args.focal_loss_gamma,
        class_weights=class_weights
    )
    criterion.to(device)

    return criterion
