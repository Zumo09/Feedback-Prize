# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
import time
from tqdm import tqdm
from typing import Iterable, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models import CriterionDETR, PrepareInputs, DETR
from datasets import FBPPostProcess


def train_one_epoch(
    tokenizer: PrepareInputs,
    model: DETR,
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
        st = time.time()

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = []
        for doc in samples:
            inputs = tokenizer([doc]).to(device)

            glob_enc_attn = torch.zeros(inputs.size()[1]).to(device)
            glob_enc_attn[0] = 1

            glob_dec_attn = torch.ones(model.num_queries).to(device)

            outputs.append(model(inputs, glob_enc_attn, glob_dec_attn))

        batch_outputs = {
            key: torch.cat([o[key] for o in outputs]) for key in outputs[0].keys()
        }

        loss_dict = criterion(batch_outputs, targets)  # type: Dict[str, torch.Tensor]

        mt = time.time()

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

        ot = time.time()

        loss_list.append(losses.item())  # type: ignore
        data_bar.set_postfix(
            {
                "lr": optimizer.param_groups[0]["lr"],
                "loss": sum(loss_list) / len(loss_list),
                "model time": f"{mt - st:.2f} s",
                "optim time": f"{ot - mt:.2f} s",
            }
        )
        if writer:
            scalars = {
                "lr": optimizer.param_groups[0]["lr"],
                "loss": losses.item(),  # type: ignore
                **loss_dict_scaled,
                **loss_dict_unscaled,
            }
            for key, value in scalars.items():
                writer.add_scalars(key, {'Train': value})


@torch.no_grad()
def evaluate(
    tokenizer: PrepareInputs,
    model: DETR,
    criterion: CriterionDETR,
    postprocessor: FBPPostProcess,
    data_loader: DataLoader,
    epoch: int,
    device: torch.device,
    tag: str = "No Tag",
    writer: Optional[SummaryWriter] = None,
):
    model.eval()
    criterion.eval()

    loss_list = []
    data_bar = tqdm(data_loader, desc=f"Valid Epoch {epoch:4d}")
    for samples, targets, infos in data_bar:
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = []
        for doc in samples:
            inputs = tokenizer([doc]).to(device)

            glob_enc_attn = torch.zeros(inputs.size()[1]).to(device)
            glob_enc_attn[0] = 1

            glob_dec_attn = torch.ones(model.num_queries).to(device)

            outputs.append(model(inputs, glob_enc_attn, glob_dec_attn))

        batch_outputs = {
            key: torch.cat([o[key] for o in outputs]) for key in outputs[0].keys()
        }

        loss_dict = criterion(batch_outputs, targets)  # type: Dict[str, torch.Tensor]
        weight_dict = criterion.weight_dict

        # loss_dict_scaled = {
        #     k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict
        # }
        # loss_dict_unscaled = {f"{k}_unscaled": v for k, v in loss_dict.items()}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)  # type: ignore

        postprocessor.add_outputs(batch_outputs, infos)

        loss_value = losses.item()  # type: ignore
        loss_list.append(loss_value)

        data_bar.set_postfix({"loss": sum(loss_list) / len(loss_list)})

    loss = sum(loss_list) / len(loss_list)
    report = postprocessor.evaluate()
    scalars = {"loss": loss, "accuracy": report["f1"]["macro_avg"]}

    if writer:
        for key, value in scalars.items():
            writer.add_scalars(key, {tag: value})

    return report
