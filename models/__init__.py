import torch
from .detr import DETR, PrepareInputs
from .matcher import HungarianMatcher
from .criterion import CriterionDETR
from transformers import LEDModel, LEDTokenizerFast  # type: ignore


def build_models(num_classes: int, args):
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
        # eos_coef=args.eos_coef,
        losses=losses,
        gamma=args.focal_loss_gamma
    )
    criterion.to(device)

    return tokenizer, model, criterion
