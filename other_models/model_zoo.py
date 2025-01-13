import torch
import torch.nn as nn
from other_models.ctvit.ctvit import CTViT
from other_models.ct_clip.ct_clip import CTCLIP
from transformers import BertModel


class CT_CLIP_enc(nn.Module):
    def __init__(self, num_classes, enc):
        super().__init__()
        self.vit = enc
        for param in self.vit.parameters():
            param.requires_grad = True
        self.head = nn.Linear(512, num_classes)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
        }
    
    def forward(self, x):
        # assume x has a shape of [N, C, D, H, W] = [N, 1, 240, 480, 480]
        z = self.vit(x, return_encoded_tokens=True)
        z = torch.mean(z, dim=(1, 2, 3))
        z = self.head(z)
        return z