#!/usr/bin/env python3
import warnings
warnings.simplefilter("ignore", UserWarning)
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode
from data_loader import *
from train import *
from model import *
from test import *
from utils import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



BATCH_SIZE = 32
EPOCHS = 10
weight_path = '../weights/vgg16_stare_final.pth'
train_folder='../Kaggle_dataset/images'
train_labels='../Kaggle_dataset/trainLabels.csv'
loading_weight_path = '../weights/vgg16_stare_test_1.pth'
kaggle_dataset = False
staredata = True
gaussian_dataset = False
TRAIN = True
TEST = True
loading_model = True

if __name__ == '__main__':
    detector = RetinoDetector()
    if kaggle_dataset:
        dataset = TotalData()
    elif staredata:
        dataset = StareData()
    else:
        assert gaussian_dataset
        dataset = GaussianData()

    if loading_model:
        print('loading model weight')
        checkpoint = torch.load(loading_weight_path)
        model = detector.model
        model.load_state_dict(checkpoint['model_state_dict'])
        device = detector.device

    train_set = TrainData(dataset.images_dir, dataset.labels)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=None)

    test_set = TestData(dataset.images_dir, dataset.labels)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=None)

    if TRAIN:
        train(train_loader, test_loader, detector, len(train_set), len(test_set), EPOCHS, weight_path, BATCH_SIZE)

    if TEST:
        checkpoint = torch.load(weight_path)
        model = detector.model
        model.load_state_dict(checkpoint['model_state_dict'])
        device = detector.device
        correct = 0.
        targets = np.zeros(len(test_set), dtype=int)
        outputs = np.zeros(len(test_set), dtype=int)

        for i, (image, label) in enumerate(test_loader):
            label = label.to(device)
            image = image.to(device)

            # compute output
            output = model(image)
            output_label = output.argmax(dim=1).cpu().numpy()
            correct_label = label.cpu().numpy()
            correct += ((output.argmax(dim=1) - label) == 0).sum().item()
            targets[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = correct_label
            outputs[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = output_label

        cm = confusion_matrix(list(targets), list(outputs))
        cm_display = ConfusionMatrixDisplay(cm)
        cm_display.plot()
        plt.show()
        print()
        print(cm)
        print("accuracy: {}".format(((targets - outputs) == 0).sum() / len(test_set)))
