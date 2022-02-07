import math
from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn.functional as F

import pandas as pd

from util import box_ops


class FBPPostProcess:
    """This module converts the model's output into the format expected by the Kaggle api"""
    def __init__(self, encoder: OrdinalEncoder, tags: pd.DataFrame) -> None:
        super().__init__()
        self.encoder = encoder
        self.tags = tags
        self.reset_results()

    def reset_results(self):
        self._results = pd.DataFrame(columns=['id', 'class', 'predictionstring'])
        self._class_results = pd.DataFrame(columns=self.encoder.categories_[0])

    def evaluate(self):
        """
        Evaluation metric defined by the Kaggle Challenge
        """

    @property
    def results(self) -> pd.DataFrame:
        """The DataFrame to be submitted for the challenge
        """
        return self._results.copy()

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
    def add_outputs(self, outputs, infos):
        """Format the outputs and save them to a dataframe
        Parameters:
            outputs: raw outputs of the model
            infos: list of dictionaries of length [batch_size] containing the length of each document of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        target_sizes = torch.stack([info["orig_len"] for info in infos], dim=0)

        assert len(out_logits) == len(target_sizes)

        no_obj_class = out_logits.size(-1) - 1

        # TODO cosa bisogna fare con i no_object??
        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)

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