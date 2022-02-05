from sklearn.preprocessing import OrdinalEncoder
import torch
from .matcher import HungarianMatcher
from .criterion import SetCriterion
from .postprocess import PostProcess

def build(args, encoder: OrdinalEncoder):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 8 # 7 + 1
    device = torch.device(args.device)

    # backbone = build_backbone(args)

    # transformer = build_transformer(args)

    # model = DETR(
    #     backbone,
    #     transformer,
    #     num_classes=num_classes,
    #     num_queries=args.num_queries,
    #     aux_loss=args.aux_loss,
    # )

    model = None

    matcher = HungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
    )

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(encoder)}

    return model, criterion, postprocessors
