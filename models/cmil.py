import torch
import numpy as np
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(self, in_chans, embed_dim, norm_layer=None, bias=True):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x shape: (batch_size, seq_length, in_chans)
        x = self.proj(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-np.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        x = x + self.pe[:x.size(1), :].unsqueeze(0).to(x.device)
        return self.dropout(x)


class FeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vits14'):
        super(FeatureExtractor, self).__init__()
        # Load pre-trained DINOv2 model from torch.hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

    def forward(self, x):
        # x shape: (batch_size, C, H, W)
        features = self.model.forward_features(x)
        # Extract the CLS token
        if isinstance(features, dict) and 'x_norm_clstoken' in features:
            cls_token = features['x_norm_clstoken']
        else:
            cls_token = features[:, 0]  # First token is the CLS token
        return cls_token  # Shape: (batch_size, embed_dim)


class SliceFusionTransformer(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, hidden_dim, num_layers, patch_size=1):
        super(SliceFusionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_chans = patch_size * embed_dim  # Compute in_chans

        # Number of patches after dividing the sequence
        self.num_patches = (seq_len + patch_size - 1) // patch_size
        self.patch_embed = PatchEmbed(in_chans=self.in_chans, embed_dim=embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=10000)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder layers with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights of the patch embedding layer and positional encoding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Pad the sequence if necessary to make it divisible by patch_size
        pad_len = (self.patch_size - seq_length % self.patch_size) % self.patch_size
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, self.embed_dim, device=x.device)
            x = torch.cat([x, padding], dim=1)  # Shape: (batch_size, new_seq_length, embed_dim)

        # Update seq_length after padding
        seq_length = x.size(1)
        num_patches = seq_length // self.patch_size  # Use a local variable

        # Reshape x to (batch_size, num_patches, patch_size, embed_dim)
        x = x.view(batch_size, num_patches, self.patch_size, self.embed_dim)
        x = x.reshape(batch_size, num_patches, -1)  # Shape: (batch_size, num_patches, patch_size * embed_dim)

        # Apply patch embedding
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch_size, 1 + num_patches, embed_dim)

        # Apply positional encoding
        x = self.positional_encoding(x)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Extract the output corresponding to the CLS token
        cls_token_output = x[:, 0, :]  # Shape: (batch_size, embed_dim)
        return cls_token_output


class CMILModel(nn.Module):
    def __init__(self, feature_extractor, transformer_model):
        super(CMILModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.transformer_model = transformer_model

    def forward(self, x):
        # x shape: (batch_size, seq_length, C, H, W)
        batch_size, seq_length, C, H, W = x.shape

        # Reshape to process slices individually
        x = x.view(batch_size * seq_length, C, H, W)
        cls_tokens = self.feature_extractor(x)  # Shape: (batch_size * seq_length, embed_dim)

        # Reshape for transformer input
        cls_tokens = cls_tokens.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, embed_dim)
        cls_token_output = self.transformer_model(cls_tokens)  # Shape: (batch_size, embed_dim)
        return cls_token_output
