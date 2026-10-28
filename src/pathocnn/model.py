"""PathoCNN: a compact CNN for colorectal histology tissue classification.

Designed from scratch by Habib Ur Rehman for PathMNIST (MedMNIST v2):
9-class classification of 28x28 H&E-stained colorectal histology patches
(NCT-CRC-HE-100K train/val; CRC-VAL-HE-7K external-center test set).

Labels (MedMNIST v2 / PathMNIST official order):
  0 adipose
  1 background
  2 debris
  3 lymphocytes
  4 mucus
  5 smooth muscle
  6 normal colon mucosa
  7 cancer-associated stroma
  8 colorectal adenocarcinoma epithelium
"""
import torch
import torch.nn as nn


class PathoCNN(nn.Module):
    """3 conv blocks (32 -> 64 -> 128 channels) + 2-layer classifier head.

    288,809 trainable parameters. Input: (N, 3, 28, 28) float tensor,
    normalized per channel with the PathMNIST train-split mean/std
    (see pathocnn.infer). The design follows the same compact
    conv-block family as the sibling SonoCNN release, re-derived for
    3-channel histology input and a 9-class head.
    """

    def __init__(self, in_ch=3, num_classes=9, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        # 28x28 -> 14x14 -> 7x7 -> 3x3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model():
    return PathoCNN()
