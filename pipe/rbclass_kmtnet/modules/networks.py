"""A script for defining neural network models for experiments.

This script defines a set of neural network models for experiments, focusing
on real/bogus classification tasks.
You can add any neural network architecture.

Reference:
    The OTrain model is inspired by Makhlouf, K., et al. "O’TRAIN: A robust
    and flexible ‘real or bogus’ classifier for the study of the optical
    transient sky." Astronomy & Astrophysics 664 (2022).
    For further details and a deeper understanding of the model's principles
    and performance, please refer to this paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class OTrain(nn.Module):
    def __init__(self, imsize=38, in_channels=3, num_classes=1):
        """
        The OTrain model, implemented in PyTorch, is inspired by the
        architecture described in the "O'Train" paper (Makhlouf, K., et al).
        For further details, please refer to the following papaer:
        - Makhlouf, K., et al. "O’TRAIN: A robust and flexible ‘real or bogus’
          classifier for the study of the optical transient sky." Astronomy &
          Astrophysics 664 (2022).
        """
        super(OTrain, self).__init__()
        self.num_classes = num_classes
        resolution = imsize
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=16,
                               kernel_size=3,
                               padding='same')
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=3,
                               padding='same')
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        resolution = resolution // 2

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding='same')
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        resolution = resolution // 2

        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        resolution = resolution // 2

        self.conv5 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               padding='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        resolution = resolution // 2

        self.in_features = resolution * resolution * 256
        self.fc1 = nn.Linear(self.in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)

        x = F.relu(self.conv3(x))
        x = self.maxpool1(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)

        x = x.view(-1, self.in_features)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.num_classes == 1:
            x = torch.sigmoid(x).squeeze(1)

        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super(ResNet18, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.backbone(x)
        if self.num_classes == 1:
            x = torch.sigmoid(x).squeeze(1)

        return x
