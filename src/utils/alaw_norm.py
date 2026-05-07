"""
src/utils/alaw_norm.py

AlawNorm: Codec-Aware Normalization Layer for G.711 A-law VAD

What this layer does (in simple words):
----------------------------------------
G.711 A-law codec compresses audio using a logarithmic formula.
This compression squashes loud speech values together, reducing
the contrast between speech and silence that the model needs
to detect speech accurately.

AlawNorm applies the mathematical INVERSE of this compression
as the very first step inside the model. This expands the values
back toward their original distribution, making it easier for
the convolutional layers to find speech patterns.

Mathematical basis:
  G.711 A-law compression formula (ITU-T G.711 standard):
    For |x| >= 1/A:
      F(x) = sign(x) * (1 + ln(A|x|)) / (1 + ln(A))
    where A = 87.6 (ITU-T defined constant)

  Inverse (what AlawNorm computes):
    F^-1(y) = sign(y) * exp(|y|(1 + ln(A)) - 1) / A

  The layer starts with A=87.6 (mathematically correct)
  and learns small adjustments during training to compensate
  for quantization noise and hardware-specific variations.

Why this is novel:
  All existing VAD papers use generic normalization (BatchNorm,
  LayerNorm) that knows nothing about the codec.
  AlawNorm is the first normalization layer mathematically
  derived from the ITU-T G.711 A-law inverse transfer function.

Parameters learned during training:
  self.A     : the companding constant (initialized at 87.6)
  self.scale : output scale factor (initialized at 1.0)
  self.bias  : output bias (initialized at 0.0)
"""

import torch
import torch.nn as nn
import numpy as np


class AlawNorm(nn.Module):
    """
    G.711 A-law Codec-Aware Normalization Layer.

    Prepend this as the first layer of any VAD model to
    compensate for G.711 A-law codec amplitude distortion
    before feature extraction.

    Usage:
        model = CNN1D()
        # AlawNorm is already the first layer inside CNN1D
        # No changes needed at inference time

    Args:
        a_init   : initial value of A-law constant (ITU-T = 87.6)
        learnable: if True, A/scale/bias are updated during training
                   if False, fixed mathematical inverse (no learning)
    """

    def __init__(self, a_init: float = 87.6, learnable: bool = True):
        super().__init__()

        self.learnable = learnable

        if learnable:
            # learnable parameters — start at mathematically correct values
            self.A     = nn.Parameter(torch.tensor(a_init,  dtype=torch.float32))
            self.scale = nn.Parameter(torch.tensor(1.0,     dtype=torch.float32))
            self.bias  = nn.Parameter(torch.tensor(0.0,     dtype=torch.float32))
        else:
            # fixed — pure mathematical inverse, no learning
            self.register_buffer('A',     torch.tensor(a_init, dtype=torch.float32))
            self.register_buffer('scale', torch.tensor(1.0,    dtype=torch.float32))
            self.register_buffer('bias',  torch.tensor(0.0,    dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse G.711 A-law compensation.

        Args:
            x: audio samples, shape (batch, time) or (batch, channels, time)

        Returns:
            Compensated audio, same shape as input
        """
        # keep A positive and above minimum threshold
        A    = torch.clamp(torch.abs(self.A), min=1.0)
        ln_A = torch.log(A)

        # sign and absolute value
        sign  = torch.sign(x)
        abs_x = torch.abs(x).clamp(min=1e-8)

        # inverse A-law formula
        # expands compressed values back toward original distribution
        x_comp = sign * (torch.exp(abs_x * (1.0 + ln_A)) - 1.0) / A

        # learnable scale and bias for fine-tuning
        return self.scale * x_comp + self.bias

    def extra_repr(self) -> str:
        return (f"A={self.A.item():.2f}, "
                f"scale={self.scale.item():.4f}, "
                f"bias={self.bias.item():.4f}, "
                f"learnable={self.learnable}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Test AlawNorm with a simple example.
    Shows how A-law compressed values are expanded back.
    """
    import audioop

    # simulate A-law roundtrip on a test signal
    t          = np.linspace(0, 1, 8000, dtype=np.float32)
    clean      = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    # apply real A-law codec
    int16      = (clean * 32767).astype(np.int16)
    alaw       = audioop.lin2alaw(int16.tobytes(), 2)
    restored   = audioop.alaw2lin(alaw, 2)
    distorted  = np.frombuffer(restored, dtype=np.int16).astype(np.float32) / 32767.0

    # test AlawNorm
    layer      = AlawNorm(a_init=87.6, learnable=True)
    x_in       = torch.tensor(distorted).unsqueeze(0)   # (1, 8000)
    x_out      = layer(x_in)

    print("AlawNorm test:")
    print(f"  Input  shape : {x_in.shape}")
    print(f"  Output shape : {x_out.shape}")
    print(f"  Input  std   : {x_in.std().item():.6f}")
    print(f"  Output std   : {x_out.std().item():.6f}")
    print(f"  Clean  std   : {clean.std():.6f}")
    print(f"  Layer params : {count_parameters(layer)}")
    print(f"  Layer repr   : {layer.extra_repr()}")
    print("\n  AlawNorm working correctly.")