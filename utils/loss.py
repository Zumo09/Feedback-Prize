from math import ceil, floor
import torch
from torch.nn import CrossEntropyLoss, Softmax
from scipy.optimize import linear_sum_assignment


def iou(y_target, y_pred, text_token):
  
    mask_pred = torch.where((text_token >= torch.floor(y_pred['center'] - y_pred['len']/2)) \
                          and (text_token <= torch.ceil(y_pred['center'] + y_pred['len']/2)))
    mask_target = torch.where((text_token >= torch.floor(y_target['center'] - y_target['len']/2)) \
                            and (text_token <= torch.ceil(y_target['center'] + y_target['len']/2)))
    
    return (mask_target * mask_pred).sum() / torch.sum(mask_target + mask_pred)


def matching_phase(y_target, y_pred, lambda_iou, lambda_l1):
  '''
  The network will always generate N object to classify, where N is the dimension
  of the last hidden size of the transformers (768). For each one will generate a probability
  to belong to each class.
  text_token just contain a vector with the indexes of the words of the text.
  It can be changed with smarter implementation.
  Assumptio to receive the pred and target as dict in the following form:
  y_pred[i] = {'class' : [s1, .. , si, ..., sn],
            'center': c,
            'len'   : l}
  '''

  text_token = torch.Tensor((range(len_sequence)), dtype=int)

  # Computation of cost matrix for hungarian algorithm
  cost_matrix = torch.zeros((last_hidden_size, num_labels+1))

  for i in range(len(y_pred)-1):
    for j in range(len(y_target)-1):
      cost_matrix[i, j] = -Softmax(y_pred[i]['class'][j]) + lambda_iou*iou(y_pred[i], y_target[j], text_token) \
                        + lambda_l1*torch.abs(y_pred[i]['center'] - y_target[j]['center'])
    
    # It can be improved with better libraries
    id_pred, id_target = linear_sum_assignment(cost_matrix)

    return zip(id_pred, id_target)


def loss(y_target, y_pred, opt_pairs):
    l = 0
    for i in range(len(y_pred)):
      id_pred = opt_pairs[i][0]
      id_target = opt_pairs[i][1]
      l += -log(Softmax(y_pred[id_pred]['class'][id_target])) + lambda_iou*iou(y_pred[id_pred], y_target[id_target], text_token) \
                        + lambda_l1*torch.abs(y_pred[id_pred]['center'] - y_target[id_target]['center'])

    return l



def DIoU_loss(pred, target):
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
