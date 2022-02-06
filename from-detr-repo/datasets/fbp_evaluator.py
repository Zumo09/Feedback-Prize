# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
FBP evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib


Guarda 
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py



"""
import os
import contextlib
import copy
import numpy as np
import torch


class FBPEvaluator(object):
    def __init__(self, fbp_gt):
        fbp_gt = copy.deepcopy(fbp_gt)
        self.fbp_gt = fbp_gt

        self.eval = {}

        self.doc_ids = []
        self.eval_docs = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.doc_ids.extend(img_ids)

        results = self.prepare_for_fbp_detection(predictions)

        # suppress pycocotools prints
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                fbp_dt = FBP.loadRes(self.fbp_gt, results) if results else FBP()
        fbp_eval = self.fbp_eval[iou_type]

        fbp_eval.fbpDt = fbp_dt
        fbp_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = evaluate(fbp_eval)

        self.eval_docs.append(eval_imgs)

    def accumulate(self):
        for fbp_eval in self.fbp_eval.values():
            fbp_eval.accumulate()

    def summarize(self):
        for iou_type, fbp_eval in self.fbp_eval.items():
            print("IoU metric: {}".format(iou_type))
            fbp_eval.summarize()

    def prepare_for_fbp_detection(self, predictions):
        fbp_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            fbp_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return fbp_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_fbp_eval(fbp_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    fbp_eval.evalImgs = eval_imgs
    fbp_eval.params.imgIds = img_ids
    fbp_eval._paramsEval = copy.deepcopy(fbp_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
