"""
Utilities for bounding box manipulation and GIoU.
"""
import torch

def box_cl_to_se(x):
    """
    Convert center-length coordinates to start-end coordinates
    """
    c, l = x.unbind(-1)
    b = [(c - 0.5 * l), (c + 0.5 * l)]
    return torch.stack(b, dim=-1)


def box_se_to_cl(x):
    """
    Convert start-end coordinates to center-length coordinates
    """
    s, e = x.unbind(-1)
    b = [(s + e) / 2, (s - e)]
    return torch.stack(b, dim=-1)

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (s, e) coordinates.

    Args:
        boxes (Tensor[N, 2]): boxes for which the area will be computed. They
            are expected to be in (s, e) format with
            ``0 <= s < e``.

    Returns:
        Tensor[N]: the area for each box
    """
    return (boxes[:, 1] - boxes[:, 0])
    

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :1], boxes2[:, :1])  # [N,M,1]
    rb = torch.min(boxes1[:, None, 1:], boxes2[:, 1:])  # [N,M,1]

    wh = (rb - lt).clamp(min=0)  # [N,M,1]
    inter = wh[:, :, 0]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

# TODO
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [s, e] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
