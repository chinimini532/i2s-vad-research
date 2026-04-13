"""
src/models/cnn1d.py

Model 1: Lightweight 1D CNN
-----------------------------
What this model does (in simple words):
  Takes 256 audio samples (32ms of audio) as input.
  Passes them through 4 convolutional layers that learn
  local patterns in the audio signal.
  Then a global average pooling squashes everything into
  a single vector, and a final linear layer outputs
  the probability of speech being present.

Why this model:
  Fastest, smallest, easiest to deploy on CM5.
  Acts as our baseline — if fancier models don't beat
  this, they are not worth the extra complexity.

Architecture:
  Input  : (batch, 1, 256)   1 channel, 256 time steps
  Conv1  : 32 filters, kernel 8
  Conv2  : 64 filters, kernel 8
  Conv3  : 128 filters, kernel 4
  Conv4  : 128 filters, kernel 4
  GAP    : global average pooling -> (batch, 128)
  FC1    : 64 units
  Output : 2 units (speech / no-speech)

Parameters: ~50K
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv1d(in_channels=1, out_channels=32,
                      kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),        # 256 -> 128

            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64,
                      kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),        # 128 -> 64

            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128,
                      kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),        # 64 -> 32

            # Block 4
            nn.Conv1d(in_channels=128, out_channels=128,
                      kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),        # 32 -> 16
        )

        # Global average pooling: averages across time dimension
        # Output: (batch, 128) regardless of input length
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, 256)        raw audio windows
        returns: (batch, num_classes) logits
        """
        # add channel dimension: (batch, 256) -> (batch, 1, 256)
        x = x.unsqueeze(1)
        x = self.features(x)        # (batch, 128, 16)
        x = self.gap(x)             # (batch, 128, 1)
        x = self.classifier(x)      # (batch, num_classes)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNN1D(num_classes=2)
    dummy = torch.randn(8, 256)     # batch of 8 windows
    out   = model(dummy)
    print(f"CNN1D")
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Params : {count_parameters(model):,}")