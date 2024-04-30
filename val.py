#!/usr/bin/env python3
import cv2
import os
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def validation(test_dataset, detector):
    device = detector.device
    model = detector.model
    correct = 0.
    fp = lambda y_pred, y: (len(y_pred) - y_pred.argmax(dim=1)[y==1].sum())
    fp_num = 0
    for i, (image, label) in enumerate(test_dataset):
        label = label.to(device)
        image = image.to(device)

        # compute output
        output = model(image)
        correct += ((output.argmax(dim=1) - label) == 0).sum().item()
        cm = confusion_matrix(list(label.cpu().detach().numpy()), list(output.argmax(dim=1).cpu().detach().numpy()))
        if len(cm) != 1:
            fp_num += cm[1][0]
        else:
            fp_num += 0
    return correct, fp_num
