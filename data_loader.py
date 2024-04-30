#!/usr/bin/env python3
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode


class StareData:
    def __init__(self):
        annotation_file = '../STARE/all-mg-codes.txt'
        images_file = '../STARE/images'
        with open(annotation_file, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        new_lines = list(map(lambda x: x.split('\t'), lines))
        self.images_dir = list(map(lambda x: os.path.join(images_file, x[0] + '.ppm'), new_lines))
        self.labels = []
        add_1 = False
        for i in range(len(new_lines)):
            if i in [46, 107, 108, 143, 166]:
                continue
            for item in new_lines[i]:
                if 'Diabetic' in item:
                    add_1 = True
            if add_1:
                self.labels.append(1)
            else:
                self.labels.append(0)
            add_1 = False
        self.images_dir.remove('../STARE/images\\im0047.ppm')
        self.images_dir.remove('../STARE/images\\im0108.ppm')
        self.images_dir.remove('../STARE/images\\im0109.ppm')
        self.images_dir.remove('../STARE/images\\im0144.ppm')
        self.images_dir.remove('../STARE/images\\im0167.ppm')
        a = 3


class TotalData:
    def __init__(self, train_folder='../Kaggle_dataset/images', train_labels='../Kaggle_dataset/trainLabels.csv'):
        print('loading data')
        self.images_dir = []
        self.labels = []
        self.folder = train_folder
        with open(train_labels, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            else:
                one_line = line.split(',')
                label = int(one_line[-1].split('\n')[0])
                self.images_dir.append(os.path.join(self.folder, one_line[0] + '.jpeg'))
                if label == 0:
                    self.labels.append(0)
                else:
                    assert label in [1, 2, 3, 4]
                    self.labels.append(1)

        print("total samples: {}".format(len(self.images_dir)))


class GaussianData:
    def __init__(self):
        annotation_file = '../archive/train.csv'
        images_file = '../archive/gaussian_filtered_images/gaussian_filtered_images'
        print('loading data')
        self.images_dir = []
        self.labels = []
        with open(annotation_file, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            else:
                one_line = line.split(',')
                label = int(one_line[-1].split('\n')[0])
                self.images_dir.append(os.path.join(images_file, one_line[0] + '.png'))
                # self.labels.append(label)
                if label == 0:
                    self.labels.append(0)
                else:
                    assert label in [1, 2, 3, 4]
                    self.labels.append(1)

        a = 3

class TrainData(Dataset):
    def __init__(self,
                 total_image,
                 total_labels):
        self.images = total_image[:int(len(total_image) * 0.8)]
        self.labels = total_labels[:int(len(total_image) * 0.8)]
        self.labels = torch.tensor(self.labels)
        print("training samples: {}".format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = cv2.imread(self.images[idx])
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        out = transforms(img)

        return out, self.labels[idx]


class TestData(Dataset):
    def __init__(self,
                 total_image,
                 total_labels):
        self.images = total_image[int(len(total_image) * 0.8):]
        self.labels = total_labels[int(len(total_labels) * 0.8):]
        print("testing samples: {}".format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # img = get_image(self.images[idx])
        img = cv2.imread(self.images[idx])
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        out = transforms(img)

        return out, self.labels[idx]

