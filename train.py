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
from val import *


def train(train_dataset, test_dataset, detector, train_length, test_length, epochs, weight_path, BATCH_SIZE):
    device = detector.device
    model = detector.model
    criterion = detector.criterion
    fp_loss = lambda y_pred, y: (len(y[y == 1]) - y_pred.argmax(dim=1)[y == 1].sum()) / len(y[y == 1])
    best_test_fp = 10000.
    print("start training")
    theta = 0.99

    for epoch in range(epochs):
        losses = []
        correct = 0.
        detector.lr_schedule(epoch)
        optimizer = detector.optimizer

        for i, (image, label) in enumerate(train_dataset):
            label = label.to(device)
            image = image.to(device)

            # compute output
            output = model(image)

            # cm = confusion_matrix(list(label.cpu().detach().numpy()), list(output.argmax(dim=1).cpu().detach().numpy()))
            # fp_loss = cm[1][0] / cm[1].sum()
            # loss = 1. * criterion(output, label) + 10. * fp_loss
            loss = criterion(output, label)
            # loss = theta * criterion(output[label == 1], label[label == 1]) \
            #     + (1 - theta) * criterion(output[label == 0], label[label == 0])

            losses.append(loss.item())

            optimizer.zero_grad()
            # loss.requires_grad = True
            loss.backward()
            optimizer.step()

            correct += ((output.argmax(dim=1) - label) == 0).sum().item()

        test_correct, test_fp = validation(test_dataset, detector)
        print('epoch: {} || loss: {} || acc: {} || val: {} || fp: {}'.format(epoch + 1,
                                                                             np.mean(losses),
                                                                             correct / train_length,
                                                                             test_correct / test_length,
                                                                             test_fp))
        if test_fp < best_test_fp:
            best_test_fp = test_fp
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, weight_path)
