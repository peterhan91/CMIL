import torch
import torch.nn as nn
from functools import partial
from models.longmae import PatchEmbed, EncoderConfig, LongNetEncoder


class VisionTransformer(nn.Module):
    def __init__(self, num_classes, img_size=(224, 416, 416), patch_size=(4, 16, 16), in_chans=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 drop_path_rate=0, checkpoint_activations=False, flash_attention=True,
                 dilated_ratio="[1, 2, 4, 8, 16]", segment_length="[768, 1536, 3072, 6144, 12288]"):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        encoder_config = EncoderConfig(
            img_size=img_size, patch_size=patch_size, vocab_size=-1,
            multiway=False, layernorm_embedding=False, normalize_output=False,
            no_output_layer=True, drop_path_rate=drop_path_rate,
            encoder_embed_dim=embed_dim, encoder_attention_heads=num_heads,
            encoder_ffn_embed_dim=int(embed_dim * mlp_ratio), encoder_layers=depth,
            checkpoint_activations=checkpoint_activations, flash_attention=flash_attention,
            dilated_ratio=dilated_ratio, segment_length=segment_length, seq_parallel=False,
        )

        self.encoder = LongNetEncoder(
            encoder_config, embed_tokens=None, embed_positions=None,
            output_projection=None, is_encoder_decoder=False
        )
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
        }

    def forward(self, x, return_feature=False):
        # x: [N, C, D, H, W]
        x = self.patch_embed(x)  # [N, L, D]
        x = x + self.pos_embed[:, 1:, :]

        # Prepend CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]
        x = self.norm(x)
        
        if return_feature:
            return x[:, 0], x[:, 1:]
        else:
            x = self.head(x[:, 0])
            return x
    

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=(256, 448, 448), patch_size=(4, 16, 16), embed_dim=768, 
        depth=12, num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        segment_length="[784, 3136, 6272, 12544, 50176]", **kwargs) 
        # 6272, 12544, 25088, 50176, 100352 MAE
    return model
