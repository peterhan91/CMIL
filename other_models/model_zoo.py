import torch
import torch.nn as nn
import monai.networks.nets as nets


def get_model(args):
    if args.model_name == 'med3d':
        from other_models.fmcib.fmcib import fmcib_model
        enc = fmcib_model(eval_mode=False, ckpt_path=None,
                    widen_factor=1, pretrained=True,
                    bias_downsample=False, conv1_t_stride=2,)
        model = fmcib_enc(num_classes=args.nb_classes, enc=enc, latent_dim=2048)

    elif args.model_name == 'fmcib':
        from other_models.fmcib.fmcib import fmcib_model
        enc = fmcib_model(eval_mode=False, ckpt_path=args.finetune)
        model = fmcib_enc(num_classes=args.nb_classes, enc=enc)
    
    elif args.model_name == 'i3d':
        from other_models.merlin import build
        enc = build.ImageEncoder(is_marlin=False)
        model = Merlin_enc(num_classes=args.nb_classes, enc=enc)

    else:
        raise ValueError(f"Model {args.model_name} not found.")
    
    return model


class GetLast(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input[-1]


class fmcib_enc(nn.Module): 
    # ResNet-50 with SSL on medical data
    def __init__(self, num_classes, enc, latent_dim=4096):
        super().__init__()
        self.cnn = enc
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        # assume x has a shape of [N, C, D, H, W] = [N, 1, 240, 480, 480]
        z = self.cnn(x)
        z = self.head(z)
        return z


class med3d_enc(nn.Module): 
    # ResNet-50 with SL on medical data
    def __init__(self, num_classes, model):
        super().__init__()
        self.model = model

        self.model.conv_seg = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet_enc(nn.Module):
    # ResNet-50 with SL on Kinetics dataset
    def __init__(self, in_ch, out_ch, 
                 spatial_dims=3, 
                 model=50,
                 ):
        super().__init__()
        
        resnet = nets.ResNetFeatures(model_name=f'resnet{model}',  spatial_dims=spatial_dims, in_channels=in_ch)
        resnet_out_ch = max([ mod.num_features for name, mod in resnet.layer4[-1]._modules.items() if "bn" in name])
        self.model = nn.Sequential(
            resnet,
            GetLast(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(resnet_out_ch, out_ch)
        )
        
    def forward(self, source):
        output = self.model(source) 
        return output


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