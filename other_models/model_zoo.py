import torch
import torch.nn as nn


class fmcib_enc(nn.Module):
    def __init__(self, num_classes, enc):
        super().__init__()
        self.cnn = enc
        self.head = nn.Linear(4096, num_classes)

    def forward(self, x):
        # assume x has a shape of [N, C, D, H, W] = [N, 1, 240, 480, 480]
        z = self.cnn(x)
        z = self.head(z)
        return z


class Merlin_enc(nn.Module):
    def __init__(self, num_classes, enc):
        super().__init__()
        self.cnn = enc
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # assume x has a shape of [N, C, D, H, W] = [N, 1, 240, 480, 480]
        z = self.cnn(x)
        z = self.head(z[0])
        return z


class CT_CLIP_enc(nn.Module):
    def __init__(self, num_classes, clip):
        super().__init__()
        self.vit = clip.visual_transformer
        self.to_visual_latent = clip.to_visual_latent
        self.relu = nn.ReLU()
        self.head = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # assume x has a shape of [N, C, D, H, W] = [N, 1, 240, 480, 480]
        z = self.vit(x, return_encoded_tokens=True)
        z = torch.mean(z, dim=1)
        z = z.view(z.shape[0], -1)
        z = self.to_visual_latent(z)
        z = self.relu(z)
        z = self.head(z)
        return z