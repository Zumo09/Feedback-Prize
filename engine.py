# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from tqdm import tqdm
from typing import Iterable, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from datasets.fbp_dataset import FBPEvaluator
from models.criterion import CriterionDETR
from models.postprocess import PostProcess


def train_one_epoch(
    model: torch.nn.Module,
    criterion: CriterionDETR,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    writer: Optional[SummaryWriter] = None,
):
    model.train()
    criterion.train()

    loss_list = []
    data_bar = tqdm(data_loader, desc=f"Train Epoch {epoch:4d}")
    for samples, targets, info in data_bar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)  # type: Dict[str, torch.Tensor]
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)  # type: ignore

        loss_dict_unscaled = {f"{k}_unscaled": v for k, v in loss_dict.items()}
        loss_dict_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict
        }
        losses_scaled = sum(loss_dict_scaled.values())  # type: ignore

        loss_value = losses_scaled.item()  # type: ignore

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()  # type: ignore
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # type: ignore
        optimizer.step()

        loss_list.append(losses.item())  # type: ignore
        data_bar.set_postfix(
            {
                "lr": optimizer.param_groups[0]["lr"],
                "mean_loss": sum(loss_list) / len(loss_list),
            }
        )
        if writer:
            scalars = {
                "lr": optimizer.param_groups[0]["lr"],
                "loss": losses.item(),  # type: ignore
                **loss_dict_scaled,
                **loss_dict_unscaled,
            }
            writer.add_scalars("Training", scalars)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: CriterionDETR,
    postprocessor: PostProcess,
    evaluator: FBPEvaluator,
    data_loader: DataLoader,
    epoch: int,
    device: torch.device,
    tag: str = "No Tag",
    writer: Optional[SummaryWriter] = None,
):
    model.eval()
    criterion.eval()

    loss_list = []
    results_accuracy = []
    data_bar = tqdm(data_loader, desc=f"Valid Epoch {epoch:4d}")
    for samples, targets, info in data_bar:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict
        }
        loss_dict_unscaled = {f"{k}_unscaled": v for k, v in loss_dict.items()}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)  # type: ignore

        orig_target_sizes = torch.stack([t["orig_len"] for t in info], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        res_acc = evaluator(results, info)
        results_accuracy.append(res_acc)

        loss_value = losses.item()  # type: ignore
        loss_list.append(loss_value)
        data_bar.set_postfix(
            {
                "mean_loss": sum(loss_list) / len(loss_list),
                "mean_accuracy": sum(results_accuracy) / len(results_accuracy),
            }
        )

        if writer:
            scalars = {
                "loss": loss_value,
                "accuracy": res_acc,
                **loss_dict_scaled,
                **loss_dict_unscaled,
            }
            writer.add_scalars(tag, scalars)
