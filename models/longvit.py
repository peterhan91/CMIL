import torch
import torch.nn as nn

from utils import trunc_normal_
from torchscale.architecture.encoder import Encoder
from torchscale.model.LongNet import LongNetEncoder
from torchscale.architecture.config import EncoderConfig


class PatchEmbed(nn.Module):
    """3D Image to Patch Embedding"""
    def __init__(self, img_size=(256, 512, 512), patch_size=(4, 16, 16), in_chans=1, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size,) * 3
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        self.img_size = img_size
        self.patch_size = patch_size
        self.D_patches = img_size[0] // patch_size[0]
        self.H_patches = img_size[1] // patch_size[1]
        self.W_patches = img_size[2] // patch_size[2]
        self.num_patches = self.D_patches * self.H_patches * self.W_patches
        print(f"Number of patches: {self.num_patches}")

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for 3D data with CLS token."""
    def __init__(self, img_size=(256, 512, 512), patch_size=32, in_chans=1, num_classes=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, flash_attention=True,
                 dilated_ratio="[1, 2, 4, 8, 16]", 
                 segment_length="[768, 1536, 3072, 6144, 12288]",
                 checkpoint_activations=False, **kwargs): 
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Embedding including CLS token
        self.pos_embed = nn.Parameter(torch.zeros(
            1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        encoder_config = EncoderConfig(
            img_size=img_size, patch_size=patch_size, vocab_size=64010,
            multiway=False, layernorm_embedding=False, normalize_output=False,
            no_output_layer=True, drop_path_rate=drop_path_rate,
            encoder_embed_dim=embed_dim, encoder_attention_heads=num_heads,
            encoder_ffn_embed_dim=int(embed_dim * mlp_ratio), encoder_layers=depth,
            checkpoint_activations=checkpoint_activations, flash_attention=flash_attention,
            dilated_ratio=dilated_ratio, segment_length=segment_length, seq_parallel=False,
        )
        if flash_attention:
            print("Using Torchscale LongNetEncoder")
            self.encoder = LongNetEncoder(
                encoder_config, embed_tokens=None, embed_positions=None,
                output_projection=None, is_encoder_decoder=False)
        else:
            print("Using Torchscale Encoder")
            self.encoder = Encoder(
                encoder_config, embed_tokens=None, embed_positions=None,
                output_projection=None, is_encoder_decoder=False)

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, D, H, W):
        N = x.shape[1]
        N_patches = N - 1  # Exclude CLS token
        N_patches_org = self.pos_embed.shape[1] - 1
        if N_patches == N_patches_org:
            # No interpolation needed
            pos_embed = self.pos_embed
        else:
            # Interpolate positional embeddings
            cls_pos_embed = self.pos_embed[:, :1, :]
            patch_pos_embed = self.pos_embed[:, 1:, :]
            dim = x.shape[-1]
            D_patches_new = D // self.patch_embed.patch_size[0]
            H_patches_new = H // self.patch_embed.patch_size[1]
            W_patches_new = W // self.patch_embed.patch_size[2]

            patch_pos_embed = patch_pos_embed.reshape(
                1, self.patch_embed.D_patches, self.patch_embed.H_patches, self.patch_embed.W_patches, dim)
            patch_pos_embed = patch_pos_embed.permute(0, 4, 1, 2, 3)
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed,
                size=(D_patches_new, H_patches_new, W_patches_new),
                mode='trilinear',
                align_corners=False,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).reshape(1, -1, dim)
            pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
        return pos_embed

    def prepare_tokens(self, x):
        B, nc, D, H, W = x.shape
        x = self.patch_embed(x)  # (B, N_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + N_patches, embed_dim)
        x = x + self.interpolate_pos_encoding(x, D, H, W)
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        x = self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]
        x = self.norm(x)
        cls_x = x[:, 0]  # Extract CLS token
        return cls_x


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
