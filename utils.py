# utils.py

import torch
import torch.nn as nn
import numpy as np


class PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(
        self,
        in_chans=1536,
        embed_dim=768,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, L, D = x.shape
        x = self.proj(x)
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, 1, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-np.log(10000.0) / embed_dim)
        )
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_length, batch_size, embed_dim)
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)
