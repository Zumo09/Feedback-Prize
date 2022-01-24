# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17EjMyXNQBafiKJLockt6ZKVBR-vJhOzt
"""

from math import ceil, floor
import torch
from torch.nn import CrossEntropyLoss, Softmax

def combined_loss(pred, target):
  '''
  DIoU = Loss_cls(target, pred) + alpha*Loss_loc(target, pred)
  Loss_cls = -log(c*_p), c*_p = exp(c_p)/sum(exp(c_p))
  Loss_loc = 1 - IoU + d(pred_center, target_center)/lambda
  Where:
    d       : the Euclidean distance;
    lambda  : normalization factor, in CV is the diagonal of the minimum enclosing rectangle that
              contains both the bounding boxes. Since here we're in a 1-D, it is the minimum length
              for containing both boxes
    
  It is derived from computer vision and modified to apply to one dimension structure. The euclidean distance doesn't present the power of two
  and consiquently also the lambda.


  '''
  pred = Softmax(pred)
  l_cls = CrossEntropyLoss(pred_class, target_class)

  text_token = torch.Tensor((range(len_sequence)), dtype=int)

  mask_pred = torch.where((text_token >= torch.floor(pred_center - pred_len/2)) and (text_token <= torch.ceil(pred_center + pred_len/2)))
  mask_target = torch.where((text_token >= torch.floor(target_center - target_len/2)) and (text_token <= torch.ceil(target_center + target_len/2)))
  iou = (mask_target * mask_pred).sum() / torch.sum(mask_target + mask_pred)

  lam = torch.max(torch.max(torch.nonzero(mask_pred),  torch.max(torch.nonzero(mask_target)))) - torch.min(torch.min(torch.nonzero(mask_pred),  torch.min(torch.nonzero(mask_target)))) + 1
  center_distance = torch.abs(pred_center - target_center)
  l_loc = 1 - iou + center_distance/lam


  return l_cls + alpha*l_loc