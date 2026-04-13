"""
src/models/transformer_vad.py

Model 4: Small Audio Transformer
-----------------------------------
What this model does (in simple words):
  Splits the 256-sample audio window into small
  patches (like tokens in a text transformer).
  Each patch becomes a "word" and self-attention
  lets every patch look at every other patch to
  understand context.

  This is the modern architecture that dominates
  NLP and is now being applied to audio.
  We use a very small version suitable for edge
  deployment.

Why this model:
  Transformer is the current state-of-the-art
  approach in most ML domains. Including it
  makes the paper more relevant to current trends.
  Reviewers expect at least one transformer comparison.

Architecture:
  Input       : (batch, 256) raw audio samples
  Patch embed : split into 16 patches of 16 samples each
  Positional  : learnable position embeddings
  Transformer : 2 encoder layers, 4 heads, dim=64
  CLS token   : classification token (like BERT)
  Output      : 2 units (speech / noise)

Parameters: ~100K
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Split audio into patches and project to embedding dim.

    Example:
      Input: 256 samples
      Patch size: 16
      Number of patches: 256/16 = 16 patches
      Each patch: 16 samples -> projected to embed_dim
    """

    def __init__(self, window_size: int = 256,
                 patch_size: int = 16,
                 embed_dim: int = 64):
        super().__init__()
        assert window_size % patch_size == 0, \
            "window_size must be divisible by patch_size"

        self.patch_size  = patch_size
        self.n_patches   = window_size // patch_size
        self.projection  = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 256)
        batch = x.shape[0]
        # reshape into patches: (batch, n_patches, patch_size)
        x = x.view(batch, self.n_patches, self.patch_size)
        # project each patch to embed_dim
        x = self.projection(x)   # (batch, n_patches, embed_dim)
        return x


class TransformerVAD(nn.Module):

    def __init__(self,
                 num_classes:  int = 2,
                 window_size:  int = 256,
                 patch_size:   int = 16,
                 embed_dim:    int = 64,
                 num_heads:    int = 4,
                 num_layers:   int = 2,
                 ff_dim:       int = 128,
                 dropout:      float = 0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(window_size, patch_size, embed_dim)
        n_patches        = window_size // patch_size

        # CLS token: one learnable vector prepended to patches
        # The final CLS representation is used for classification
        # (same idea as BERT's [CLS] token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # positional embeddings: tell the model where each patch is
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim)  # +1 for CLS
        )
        self.pos_drop  = nn.Dropout(dropout)

        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,    # (batch, seq, dim) format
            norm_first=True,     # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, 256)
        returns: (batch, num_classes)
        """
        batch = x.shape[0]

        # patch embedding
        x = self.patch_embed(x)          # (batch, n_patches, embed_dim)

        # prepend CLS token
        cls = self.cls_token.expand(batch, -1, -1)  # (batch, 1, embed_dim)
        x   = torch.cat([cls, x], dim=1)            # (batch, n_patches+1, embed_dim)

        # add positional embedding
        x = self.pos_drop(x + self.pos_embed)

        # transformer encoder
        x = self.transformer(x)          # (batch, n_patches+1, embed_dim)
        x = self.norm(x)

        # take CLS token output for classification
        cls_out = x[:, 0]                # (batch, embed_dim)

        return self.classifier(cls_out)  # (batch, num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransformerVAD(num_classes=2)
    dummy = torch.randn(8, 256)
    out   = model(dummy)
    print(f"TransformerVAD")
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Params : {count_parameters(model):,}")