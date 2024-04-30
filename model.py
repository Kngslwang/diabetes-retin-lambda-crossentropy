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


class RetinoDetector():
    def __init__(self, class_num=2, start_lr=0.001):
        super().__init__()
        self.lr = start_lr
        self.class_num = class_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model_init()
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def model_init(self):
        # model_ft = models.alexnet().to(self.device)
        model_ft = models.vgg16(weights='IMAGENET1K_V1').to(self.device)
        for param in model_ft.parameters():
            param.requires_grad = False

        n_inputs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(n_inputs, self.class_num)
        model_ft.classifier.append(nn.Softmax())
        # model = LeNet().to(self.device)
        model_ft = model_ft.to(self.device)
        return model_ft

    def lr_schedule(self, epoch):
        if epoch < 10:
            self.lr = self.lr
        elif epoch < 20:
            self.lr = self.lr / 10
        elif epoch < 30:
            self.lr = self.lr / 10
        elif epoch < 40:
            self.lr = self.lr / 10
        else:
            self.lr = self.lr / 10
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))