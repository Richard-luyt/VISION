import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import models


class FusionModel(nn.Module):
    def __init__(self):
        # T, C, H, W
        super(FusionModel, self).__init__()
        resnet = models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        hidden_dim = 512
        self.LSTM = nn.LSTM(
            input_size=hidden_dim, hidden_size=128, num_layers=2, batch_first=True
        )
        self.regressor = nn.Linear(128, 4)

    def forward(self, x):
        # Batch T C H W
        B, T, C, H, W = x.size()
        input = x.reshape((B * T, C, H, W))
        X = self.backbone(input)
        X = X.reshape(X.size(0), -1)
        X = X.reshape(B, T, -1)
        output, _ = self.LSTM(X)
        # B, T, Hidden size
        final = output[:, -1, :]
        predict = self.regressor(final)
        return predict


class FusionModel2(nn.Module):
    def __init__(self):
        # T, C, H, W
        super(FusionModel2, self).__init__()
        resnet = models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.spatial_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.intersect = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(p=0.3)
        )
        hidden_dim = 256
        self.LSTM = nn.LSTM(
            input_size=hidden_dim, hidden_size=256, num_layers=2, batch_first=True
        )
        self.regressor = nn.Linear(256, 4)

    def forward(self, x):
        # Batch T C H W
        B, T, C, H, W = x.size()
        input = x.reshape((B * T, C, H, W))
        X = self.backbone(input)
        # B*T 512 7 7
        X = self.spatial_pool(X)
        # B*T 512 2 2
        X = X.reshape(X.size(0), -1)
        # B*T 2048
        X = self.intersect(X)
        # B*T 128
        X = X.reshape(B, T, -1)
        # B T 128
        output, _ = self.LSTM(X)
        # B T 128
        final = output[:, -1, :]
        predict = self.regressor(final)
        return predict
