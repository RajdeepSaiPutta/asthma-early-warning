"""small cnn on log mel spectrograms for icbhi recording classification.

input: (batch, 1, n_mels, time). output: logits over 4 classes.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectrogramCNN(nn.Module):
    def __init__(self, n_mels: int = 64, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).flatten(1)
        return self.head(z)
