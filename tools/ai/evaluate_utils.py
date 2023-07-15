import numpy as np
import torch

from tools.general.json_utils import read_json


def calculate_for_tags(pred_tags, gt_tags):
    """This function calculates precision, recall, and f1-score using tags.

    Args:
        pred_tags: 
            The type of variable is list.
            The type of each element is string.

        gt_tags:
            The type of variable is list.
            the type of each element is string.

    Returns:
        precision:
            pass

        recall:
            pass

        f1-score:
            pass
    """
    if len(pred_tags) == 0 and len(gt_tags) == 0:
        return 100, 100, 100
    elif len(pred_tags) == 0 or len(gt_tags) == 0:
        return 0, 0, 0

    pred_tags = np.asarray(pred_tags)
    gt_tags = np.asarray(gt_tags)

    precision = pred_tags[:, np.newaxis] == gt_tags[np.newaxis, :]
    recall = gt_tags[:, np.newaxis] == pred_tags[np.newaxis, :]

    precision = np.sum(precision) / len(precision) * 100
    recall = np.sum(recall) / len(recall) * 100

    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1_score


def calculate_mIoU(pred_mask, gt_mask):
    """This function is to calculate precision, recall, and f1-score using tags.

    Args:
        pred_mask: 
            The type of variable is numpy array.

        gt_mask:
            The type of variable is numpy array.

    Returns:
        miou:
            miou is meanIU.
    """
    inter = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    epsilon = 1e-5
    miou = (np.sum(inter) + epsilon) / (np.sum(union) + epsilon)
    return miou * 100


class Calculator_For_mIoU:
    def __init__(self, class_num):
        self.classes = class_num

        self.clear()

    def get_data(self, pred_mask, gt_mask):
        obj_mask = gt_mask < 255
        correct_mask = (pred_mask == gt_mask) * obj_mask

        P_list, T_list, TP_list = [], [], []
        for i in range(self.classes):
            P_list.append(np.sum((pred_mask == i)*obj_mask))
            T_list.append(np.sum((gt_mask == i)*obj_mask))
            TP_list.append(np.sum((gt_mask == i)*correct_mask))

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.classes):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask < 255
        correct_mask = (pred_mask == gt_mask) * obj_mask

        for i in range(self.classes):
            self.P[i] += np.sum((pred_mask == i)*obj_mask)
            self.T[i] += np.sum((gt_mask == i)*obj_mask)
            self.TP[i] += np.sum((gt_mask == i)*correct_mask)

    def get(self, clear=True):
        IoU_dic = {}
        IoU_list = []

        FP_list = []  # over activation
        FN_list = []  # under activation

        for i in range(self.classes):
            IoU = self.TP[i]/(self.T[i]+self.P[i]-self.TP[i]+1e-10) * 100
            FP = (self.P[i]-self.TP[i]) / \
                (self.T[i] + self.P[i] - self.TP[i] + 1e-10)
            FN = (self.T[i]-self.TP[i]) / \
                (self.T[i] + self.P[i] - self.TP[i] + 1e-10)

            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)

        mIoU = np.mean(np.asarray(IoU_list))
        mIoU_foreground = np.mean(np.asarray(IoU_list)[1:])

        FP = np.mean(np.asarray(FP_list))
        FN = np.mean(np.asarray(FN_list))

        if clear:
            self.clear()

        return mIoU, mIoU_foreground

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []

        for _ in range(self.classes):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # mask = (label_true >= 0) & (label_true < self.num_classes)
        mask = (label_true >= 0) & (label_true < self.num_classes) & (
            label_pred < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        # recall = np.nanmean(recall)
        precision = np.diag(self.hist) / self.hist.sum(axis=0)
        # precision = np.nanmean(precision)
        TP = np.diag(self.hist)
        TN = self.hist.sum(axis=1) - np.diag(self.hist)
        FP = self.hist.sum(axis=0) - np.diag(self.hist)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) +
                                   self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        return acc, recall, precision, TP, TN, FP, cls_iu, mean_iu, fwavacc


class IOUMetric2:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = torch.zeros((num_classes, num_classes)).cuda()

    def _fast_hist(self, label_pred, label_true):
        # mask = (label_true >= 0) & (label_true < self.num_classes)
        mask = (label_true >= 0) & (label_true < self.num_classes) & (
            label_pred < self.num_classes)
        hist = torch.bincount(
            self.num_classes * label_true[mask].long() +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = torch.diag(self.hist) / (self.hist.sum(axis=1) +
                                      self.hist.sum(axis=0) - torch.diag(self.hist))
        mean_iu = np.nanmean(iu.cpu().numpy())

        return mean_iu, mean_iu
