"""
src/models/ecapa_vad.py

Model 3: ECAPA-TDNN inspired VAD
-----------------------------------
What this model does (in simple words):
  ECAPA-TDNN is a state-of-the-art architecture
  originally designed for speaker recognition.
  It uses Squeeze-and-Excitation (SE) blocks which
  let the model learn WHICH channels/features are
  most important and amplify them.

  Think of it like an attention mechanism but for
  channels instead of time steps.

  We adapt it here for binary VAD classification.
  It is the most powerful of our three models but
  also the largest.

Why this model:
  Represents the "state of the art" comparison in
  our paper. If our lightweight CNN1D gets close to
  ECAPA performance at much lower latency, that is
  a strong result for edge deployment.

Architecture:
  Input       : (batch, 1, 256)
  Conv layer  : 512 channels
  3x SE-Res2 blocks with different dilations
  Aggregation : attentive statistics pooling
  Output      : 2 units

Parameters: ~300K (kept small for edge deployment)

Note: This is our own lightweight ECAPA-inspired
implementation, not the full SpeechBrain pretrained
model (which is too large for edge deployment).
We keep the key ideas: SE blocks + multi-scale
convolutions + attentive pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze and Excitation block.

    Squeeze: global average pool -> single value per channel
    Excitation: small FC network learns channel importance weights
    Scale: multiply each channel by its learned weight

    This lets the model focus on the most useful feature channels
    and suppress less useful ones.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.squeeze    = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        s = self.squeeze(x)                          # (batch, channels, 1)
        e = self.excitation(s)                       # (batch, channels)
        e = e.unsqueeze(-1)                          # (batch, channels, 1)
        return x * e                                 # scale each channel


class SERes2Block(nn.Module):
    """
    SE + Res2 block.
    Res2: splits channels into groups, processes at different scales,
    then combines. Captures multi-scale temporal patterns.
    """

    def __init__(self, channels: int, kernel_size: int,
                 dilation: int, scale: int = 4):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # multi-scale conv
        width = channels // scale
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=dilation * (kernel_size - 1) // 2),
                nn.BatchNorm1d(width),
                nn.ReLU(),
            )
            for _ in range(scale - 1)
        ])
        self.scale  = scale
        self.width  = width

        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        self.se      = SEBlock(channels)
        self.relu    = nn.ReLU()

        # residual projection if needed
        self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)

        # split into scale groups
        chunks = torch.chunk(x, self.scale, dim=1)
        outs   = [chunks[0]]
        for i, conv in enumerate(self.convs):
            if i == 0:
                outs.append(conv(chunks[i + 1]))
            else:
                outs.append(conv(chunks[i + 1] + outs[-1]))
        x = torch.cat(outs, dim=1)

        x = self.conv2(x)
        x = self.se(x)
        x = self.relu(x + residual)
        return x


class AttentiveStatsPooling(nn.Module):
    """
    Attentive statistics pooling.
    Instead of simple average pooling, learns WHICH time
    steps are most important and weights them accordingly.
    Returns concatenated weighted mean and std.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        w    = self.attention(x)                     # attention weights
        mean = (w * x).sum(dim=2)                    # weighted mean
        var  = (w * x ** 2).sum(dim=2) - mean ** 2  # weighted variance
        std  = torch.sqrt(var.clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)         # (batch, 2*channels)


class ECAPAVAD(nn.Module):

    def __init__(self, num_classes: int = 2, channels: int = 128):
        super().__init__()

        # initial conv
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # three SE-Res2 blocks with increasing dilation
        self.block1 = SERes2Block(channels, kernel_size=3, dilation=1)
        self.block2 = SERes2Block(channels, kernel_size=3, dilation=2)
        self.block3 = SERes2Block(channels, kernel_size=3, dilation=4)

        # aggregate multi-scale features
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        # attentive pooling
        self.asp = AttentiveStatsPooling(channels)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, 256)
        returns: (batch, num_classes)
        """
        x  = x.unsqueeze(1)                # (batch, 1, 256)
        x  = self.input_conv(x)            # (batch, C, 256)

        x1 = self.block1(x)                # (batch, C, 256)
        x2 = self.block2(x1)               # (batch, C, 256)
        x3 = self.block3(x2)               # (batch, C, 256)

        # concatenate outputs from all blocks
        x  = torch.cat([x1, x2, x3], dim=1)  # (batch, 3C, 256)
        x  = self.mfa(x)                      # (batch, C, 256)

        x  = self.asp(x)                   # (batch, 2C)
        x  = self.classifier(x)            # (batch, num_classes)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ECAPAVAD(num_classes=2)
    dummy = torch.randn(8, 256)
    out   = model(dummy)
    print(f"ECAPAVAD")
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Params : {count_parameters(model):,}")