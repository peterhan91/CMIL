import torch
import torch.nn as nn
import numpy as np
from functools import partial

from utils import trunc_normal_
from torchscale.architecture.config import EncoderConfig
from torchscale.model.LongNet import LongNetEncoder
from models.longvit import PatchEmbed


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: tuple of int (D, H, W)
    Return:
    pos_embed: [D*H*W, embed_dim] or [1+D*H*W, embed_dim] (w/ or w/o cls_token)
    """
    D, H, W = grid_size
    grid_d = np.arange(D, dtype=np.float32)
    grid_h = np.arange(H, dtype=np.float32)
    grid_w = np.arange(W, dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, -1])

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 6 == 0, "Embed dimension must be divisible by 6"
    c = embed_dim // 3
    pos_embed = []
    for i in range(3):
        pe = get_1d_sincos_pos_embed_from_grid(c, grid[i])
        pos_embed.append(pe)
    pos_embed = np.concatenate(pos_embed, axis=1)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: int
    pos: numpy array of positions (M,)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class MaskedAutoencoderViT3D(nn.Module):
    """Masked Autoencoder with Vision Transformer backbone for 3D data using LongNetEncoder and LongNetDecoder."""
    def __init__(self, img_size=(224, 416, 416), patch_size=(4, 16, 16), in_chans=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 dilated_ratio="[1, 2, 4, 8, 16]",
                 segment_length="[768, 1536, 3072, 6144, 12288]",
                 checkpoint_activations=False,
                 flash_attention=True,
                 drop_path_rate=0.,
                 **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                    in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed sin-cos positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # Encoder configuration using EncoderConfig
        encoder_config = EncoderConfig(
            img_size=img_size, patch_size=patch_size, vocab_size=64010,
            multiway=False, layernorm_embedding=False, normalize_output=False,
            no_output_layer=True, drop_path_rate=drop_path_rate,
            encoder_embed_dim=embed_dim, encoder_attention_heads=num_heads,
            encoder_ffn_embed_dim=int(embed_dim * mlp_ratio), encoder_layers=depth,
            checkpoint_activations=checkpoint_activations, flash_attention=flash_attention,
            dilated_ratio=dilated_ratio, segment_length=segment_length, seq_parallel=False,
        )

        # Using LongNetEncoder
        self.encoder = LongNetEncoder(
            encoder_config, embed_tokens=None, embed_positions=None,
            output_projection=None, is_encoder_decoder=False)

        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        # Decoder configuration using DecoderConfig
        decoder_config = EncoderConfig(
            img_size=img_size, patch_size=patch_size,
            vocab_size=64010, multiway=False, layernorm_embedding=False,
            normalize_output=False, no_output_layer=True, drop_path_rate=drop_path_rate,
            encoder_embed_dim=decoder_embed_dim, encoder_attention_heads=decoder_num_heads,
            encoder_ffn_embed_dim=int(decoder_embed_dim * mlp_ratio), encoder_layers=decoder_depth,
            checkpoint_activations=checkpoint_activations, flash_attention=flash_attention,
            dilated_ratio=dilated_ratio, segment_length=segment_length, seq_parallel=False
        )

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        # Using LongNetDecoder
        self.decoder = LongNetEncoder(
            decoder_config, embed_tokens=None, embed_positions=None,
            output_projection=None, is_encoder_decoder=False)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        patch_size_product = patch_size[0] * patch_size[1] * patch_size[2]
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size_product * in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize and freeze positional embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            (self.patch_embed.D_patches, self.patch_embed.H_patches, self.patch_embed.W_patches),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            (self.patch_embed.D_patches, self.patch_embed.H_patches, self.patch_embed.W_patches),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        # Initialize layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Weight initialization
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, D, H, W)
        x: (N, L, patch_size_D * patch_size_H * patch_size_W * C)
        """
        p = self.patch_embed.patch_size  # (p_d, p_h, p_w)
        assert imgs.shape[2] % p[0] == 0 and imgs.shape[3] % p[1] == 0 and imgs.shape[4] % p[2] == 0

        d = imgs.shape[2] // p[0]
        h = imgs.shape[3] // p[1]
        w = imgs.shape[4] // p[2]
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], d, p[0], h, p[1], w, p[2]))
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # N, d, h, w, p_d, p_h, p_w, C
        x = x.reshape(shape=(imgs.shape[0], d * h * w, p[0] * p[1] * p[2] * imgs.shape[1]))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, p_d * p_h * p_w * C)
        imgs: (N, C, D, H, W)
        """
        p = self.patch_embed.patch_size
        c = self.patch_embed.proj.in_channels
        d = self.patch_embed.D_patches
        h = self.patch_embed.H_patches
        w = self.patch_embed.W_patches
        x = x.reshape(shape=(x.shape[0], d, h, w, p[0], p[1], p[2], c))
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # N, C, D, p_d, H, p_h, W, p_w
        x = x.reshape(shape=(x.shape[0], c, d * p[0], h * p[1], w * p[2]))
        return x

    def random_masking(self, x, mask_ratio):
        """
        Random masking of patches.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,D))

        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # Embed patches
        x = self.patch_embed(x)  # (N, L, D)
        # Add positional encoding
        x = x + self.pos_embed[:, 1:, :]

        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Append class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        # Pass through encoder
        x = self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)

        # Prepare mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Exclude cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[2]))  # Unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Re-append cls token

        # Add positional encoding
        x = x + self.decoder_pos_embed

        # Pass through decoder
        x = self.decoder(src_tokens=None, token_embeddings=x)["encoder_out"]
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, D, H, W]
        pred: [N, L, p_d*p_h*p_w*C]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """
        Forward pass
        imgs: [N, C, D, H, W]
        mask_ratio: float, percentage of patches to mask
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p_d*p_h*p_w*C]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT3D(
        img_size=(128, 224, 224), patch_size=(4, 16, 16), embed_dim=384, 
        depth=12, num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=16, 
        segment_length="[392, 784, 1568, 3136, 6272]",
        **kwargs)
    return model


mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b


if __name__ == '__main__':
    model = mae_vit_small_patch16().half().to('cuda:1')
    imgs = torch.randn(32, 1, 256, 416, 416).half().to('cuda:1')
    loss, pred, mask = model(imgs)
    print(loss)
    print(pred.shape)
    print(mask.shape)