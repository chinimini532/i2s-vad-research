import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.alaw_norm import AlawNorm
"""
src/models/wavenet_small.py

Model 2: Small WaveNet-style Dilated CNN
------------------------------------------
What this model does (in simple words):
  Same idea as CNN1D but uses DILATED convolutions.
  Dilation means the convolution skips samples —
  so it can see a much wider time range without
  needing more parameters.

  Example with dilation=4, kernel=3:
    Normal conv looks at positions: [0, 1, 2]
    Dilated conv looks at positions: [0, 4, 8]
  So it captures longer-range patterns in the audio.

  This is especially useful for speech because
  phonemes and syllables span 20-100ms, and
  dilated convolutions can capture that range
  efficiently.

Why this model:
  Better at capturing temporal rhythm of speech
  vs noise. Should outperform CNN1D on cases
  where the difference between speech and noise
  is in the rhythm/pattern rather than just energy.

Architecture:
  Input       : (batch, 1, 256)
  Initial conv: 64 filters
  4 dilated residual blocks (dilation = 1,2,4,8)
  GAP         : global average pooling
  Output      : 2 units

Parameters: ~150K
"""

import torch
import torch.nn as nn


class DilatedResBlock(nn.Module):
    """
    One residual block with dilated convolution.

    Residual connection means: output = conv(input) + input
    This helps gradients flow during training and
    lets the model learn small corrections rather
    than complete transformations.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.alaw_norm = AlawNorm()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=dilation,       # keeps same length
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,          # 1x1 conv to mix channels
            ),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alaw_norm(x)
        return self.relu(self.conv(x) + x)   # residual connection


class WaveNetSmall(nn.Module):

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # initial projection: 1 channel -> 64 channels
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # dilated residual blocks
        # dilation doubles each block: 1, 2, 4, 8
        # receptive field grows exponentially
        self.res_blocks = nn.Sequential(
            DilatedResBlock(channels=64, dilation=1),
            DilatedResBlock(channels=64, dilation=2),
            DilatedResBlock(channels=64, dilation=4),
            DilatedResBlock(channels=64, dilation=8),
        )

        # downsample before pooling
        self.downsample = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

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
        x shape: (batch, 256)
        returns: (batch, num_classes)
        """
        x = x.unsqueeze(1)              # (batch, 1, 256)
        x = self.input_conv(x)          # (batch, 64, 256)
        x = self.res_blocks(x)          # (batch, 64, 256)
        x = self.downsample(x)          # (batch, 128, 64)
        x = self.gap(x)                 # (batch, 128, 1)
        x = self.classifier(x)          # (batch, num_classes)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = WaveNetSmall(num_classes=2)
    dummy = torch.randn(8, 256)
    out   = model(dummy)
    print(f"WaveNetSmall")
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Params : {count_parameters(model):,}")