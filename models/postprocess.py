import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the Kaggle api"""

    @staticmethod
    def prediction_string(start, end):
        return " ".join(str(i) for i in range(math.floor(start), math.ceil(end)))

    @staticmethod
    def filter_no_object(scores, labels, boxes, no_obj_class):
        fs = []
        fl = []
        fb = []

        for s, l, b in zip(scores, labels, boxes):
            if l != no_obj_class:
                fs.append(s)
                fl.append(l)
                fb.append(b)

        return fs, fl, fb

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

        no_obj_class = out_logits.size(-1) - 1

        # TODO cosa bisogna fare con i no_object??
        prob = F.softmax(out_logits, -1)
        print(prob.size())
        print(prob[..., :-1].size())
        scores, labels = prob[..., :-1].max(-1)

        # convert to [start, end] format
        boxes = box_ops.box_cl_to_se(out_bbox)
        # and from relative [0, 1] to absolute [0, tarx_len] coordinates
        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for s, l, b in zip(scores, labels, boxes):
            s, l, b = self.filter_no_object(s, l, b, no_obj_class)
            results.append(
                {
                    "scores": s,
                    "labels": l,
                    "boxes": b,
                    "prediction_strings": [
                        self.prediction_string(start, end) for start, end in b
                    ],
                }
            )

        return results
