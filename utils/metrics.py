from functools import partial
import numpy as np
import torch

class LossMetric():
    def __init__(self):
        self.avg = 0
        self.n = 0
    
    def update(self, value):
        self.n += 1
        self.avg = (value + self.avg * (self.n - 1)) / self.n

    def get(self):
        return self.avg

class IoUMetric():
    def __init__(self):
        self.avg = 0
        self.n = 0
        self.thresholds = np.arange(0.5, 0.95, 0.05)
    
    def update(self, y_pred, y_true):
        self.n += 1
        partial_iou = torch.zeros((y_true.shape[0]))
        for th in self.thresholds:
            y_pred_th = (y_pred > th).type(torch.int8)
            tp = torch.sum(torch.logical_and(y_pred_th[:,]==1, y_true[:,]==1), axis=1)
            fp = torch.sum(torch.logical_and(y_pred_th[:,]==1, y_true[:,]==0), axis=1)
            fn = torch.sum(torch.logical_and(y_pred_th[:,]==0, y_true[:,]==1), axis=1)
            partial_iou += tp / (fp + tp + fn)
        partial_iou /= len(self.thresholds)
        value = torch.mean(partial_iou)
        self.avg = (value + self.avg * (self.n - 1)) / self.n

    def get(self):
        return self.avg
 