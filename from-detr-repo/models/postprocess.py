import math
from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the Kaggle api"""

    def __init__(self, encoder: OrdinalEncoder):
        self.encoder = encoder

    @staticmethod
    def prediction_string(start, end):
        return " ".join(str(i) for i in range(math.floor(start), math.ceil(end)))

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the length of each document of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 1

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [start, end] format
        boxes = box_ops.box_cl_to_se(out_bbox)
        # and from relative [0, 1] to absolute [0, tarx_len] coordinates
        text_len = target_sizes.unbind(1)
        scale_fct = torch.stack([text_len, text_len], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        labels = self.encoder.inverse_transform(labels)
        results = [
            {
                "scores": s,
                "labels": l,
                "boxes": b,
                "prediction_strings": [
                    self.prediction_string(start, end) for start, end in b
                ],
            }
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results
