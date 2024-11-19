import torch
import torch.nn as nn

from functools import partial
from models.longvit import VisionTransformer
from torchscale.architecture.config import EncoderConfig
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def get_vit_config(
        img_size=1024, patch_size=32, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=False,
        layernorm_embedding=False, normalize_output=False, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=384, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(384 * mlp_ratio), encoder_layers=12,
        checkpoint_activations=checkpoint_activations,
    )


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class LongViTForClassification(nn.Module):
    def __init__(
            self, 
            args,
            num_classes, 
            norm_layer=nn.LayerNorm, 
            seq_parallel=False,
            **kwargs
    ):
        super().__init__()
        self.model = VisionTransformer(
                        img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.encoder_embed_dim, 
                        depth=args.encoder_layers, num_heads=args.encoder_attention_heads, 
                        mlp_ratio=4, drop_path_rate=args.drop_path_rate,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                        checkpoint_activations=args.checkpoint_activations, seq_parallel=seq_parallel
                    )
        embed_dim = args.encoder_embed_dim
        self.depth = args.encoder_layers
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.fc_norm.apply(self._init_weights)
        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return self.depth

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.pos_embed'}
    
    def forward(self, image, **kwargs):
        x = self.model(image)
        t = x[:, :]
        cls_x = self.fc_norm(t)
        return self.head(cls_x)