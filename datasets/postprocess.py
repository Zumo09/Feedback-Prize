import math
from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn.functional as F

import pandas as pd

from util import box_ops


class FBPPostProcess:
    """This module converts the model's output into the format expected by the Kaggle api"""
    def __init__(self, encoder: OrdinalEncoder, tags: pd.DataFrame, no_obj_class: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.tags = tags
        self.no_obj_class = no_obj_class
        self.reset_results()

    def reset_results(self):
        self._results = []

    def evaluate(self):
        """
        Evaluation metric defined by the Kaggle Challenge
        """

    @property
    def results(self) -> pd.DataFrame:
        """The DataFrame to be submitted for the challenge
        """
        if len(self._results) == 0:
            return pd.DataFrame(columns=['id', 'class', 'predictionstring', 'score'])
        return pd.DataFrame(self._results)

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
        target_sizes = torch.Tensor([info["length"] for info in infos])

        assert len(out_logits) == len(target_sizes)

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)

        # from relative [0, 1] to absolute [0, tarx_len] coordinates
        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = out_bbox * scale_fct[:, None, :]
        # and convert to [start, end] format
        boxes = box_ops.box_cl_to_se(boxes)
        # round and positive
        boxes = torch.round(boxes)
        boxes = torch.relu(boxes).int()

        for i, l, s, b in zip(infos, labels, scores, boxes):
            self._add(i, l, s, b)


    def _add(self, info, labels, scores, boxes):
        doc_id = info['id']
        for l, s, b in zip(labels, scores, boxes):
            if l != self.no_obj_class:
                l = self.encoder.inverse_transform(l.reshape(-1, 1))
                self._results.append({
                    'id': doc_id,
                    'class': l[0][0],
                    'predictionstring': self.prediction_string(b),
                    'score': s.item()
                })
    
    @staticmethod
    def prediction_string(box):
        start, end = box
        return " ".join(str(i) for i in range(start, end))