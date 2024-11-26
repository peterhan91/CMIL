import torch
from torch import nn
import torch.nn.functional as F
import random
import math
import numpy as np
from collections.abc import Sequence


import torch
from torch import nn
import random
from collections.abc import Sequence

class RandomCrop3D(nn.Module):
    """Crop a random portion of a 3D image.

    Args:
        size (int or sequence): Desired output size of the crop (d, h, w).
            If size is an int, a cubic output size (size, size, size) is made.
    """

    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size, size)
        elif isinstance(size, Sequence) and len(size) == 3:
            self.size = size
        else:
            raise ValueError("Size should be an int or a sequence of length 3.")

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for random crop.

        Args:
            img (Tensor): Input 3D image tensor.
            output_size (tuple): Desired output size of the crop.

        Returns:
            tuple: Parameters (i, j, k) for the crop.
        """
        depth, height, width = img.shape[-3:]
        d, h, w = output_size
        if depth < d or height < h or width < w:
            raise ValueError("Requested crop size is bigger than input size.")

        i = random.randint(0, depth - d)
        j = random.randint(0, height - h)
        k = random.randint(0, width - w)
        return i, j, k

    def forward(self, img):
        """
        Args:
            img (Tensor): 3D image to be cropped.

        Returns:
            Tensor: Randomly cropped 3D image.
        """
        i, j, k = self.get_params(img, self.size)
        d, h, w = self.size
        img = img[..., i:i + d, j:j + h, k:k + w]
        return img



class RandomResizedCrop3D(nn.Module):
    """Crop a random portion of a 3D image and resize it to a given size.

    Args:
        size (int or sequence): Expected output size of the crop for each edge (d, h, w).
            If size is an int, a cubic output size (size, size, size) is made.
        scale (tuple of float): Range of volume scale of the cropped area before resizing.
        ratio (tuple of float): Range of aspect ratio of the cropped area before resizing.
        interpolation (str): Desired interpolation mode. Default is `'trilinear'`.
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation='trilinear',
    ):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size, size)
        elif isinstance(size, Sequence) and len(size) == 3:
            self.size = size
        else:
            raise ValueError("Size should be an int or a sequence of length 3.")

        if not (isinstance(scale, Sequence) and len(scale) == 2):
            raise ValueError("Scale should be a sequence of length 2.")
        if not (isinstance(ratio, Sequence) and len(ratio) == 2):
            raise ValueError("Ratio should be a sequence of length 2.")

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for a random sized crop.

        Args:
            img (Tensor): Input 3D image tensor.
            scale (tuple): Range of volume scale of the crop.
            ratio (tuple): Range of aspect ratio of the crop.

        Returns:
            tuple: Parameters (i, j, k, d, h, w) for the crop.
        """
        depth, height, width = img.shape[-3:]
        area = depth * height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_volume = area * random.uniform(scale[0], scale[1])
            aspect = math.exp(random.uniform(log_ratio[0], log_ratio[1]))

            # Compute dimensions of the crop
            d = int(round((target_volume * aspect) ** (1/3)))
            h = int(round((target_volume / aspect) ** (1/3)))
            w = int(round((target_volume / aspect) ** (1/3)))

            if d <= depth and h <= height and w <= width:
                i = random.randint(0, depth - d)
                j = random.randint(0, height - h)
                k = random.randint(0, width - w)
                return i, j, k, d, h, w

        # Fallback to center crop
        d = min(depth, int(round(depth * scale[1])))
        h = min(height, int(round(height * scale[1])))
        w = min(width, int(round(width * scale[1])))
        i = (depth - d) // 2
        j = (height - h) // 2
        k = (width - w) // 2
        return i, j, k, d, h, w

    def forward(self, img):
        """
        Args:
            img (Tensor): 3D image to be cropped and resized.

        Returns:
            Tensor: Randomly cropped and resized 3D image.
        """
        i, j, k, d, h, w = self.get_params(img, self.scale, self.ratio)
        img = img[..., i:i + d, j:j + h, k:k + w]

        # Prepare for interpolation
        if img.dim() == 3:
            # Shape: (D, H, W)
            img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif img.dim() == 4:
            # Shape: (C, D, H, W)
            img = img.unsqueeze(0)  # Add batch dimension
        elif img.dim() == 5:
            # Shape: (N, C, D, H, W)
            pass
        else:
            raise ValueError("Unsupported input shape.")

        # Resize the image
        img = F.interpolate(
            img,
            size=self.size,
            mode=self.interpolation,
            align_corners=False
        )

        # Remove added dimensions if necessary
        if img.shape[0] == 1 and img.dim() > 4:
            img = img.squeeze(0)

        return img


class ToTensor3D:
    """Convert a NumPy ndarray (D x H x W) to a torch.FloatTensor of shape (1 x D x H x W)."""

    def __call__(self, pic):
        """
        Args:
            pic (numpy.ndarray): 3D image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if not isinstance(pic, np.ndarray):
            raise TypeError(f'Input pic must be a NumPy ndarray, but got {type(pic)}.')
        if pic.ndim != 3:
            raise ValueError(f'Input pic must be a 3D array, but got array with {pic.ndim} dimensions.')

        # Convert the NumPy array to a torch tensor
        img = torch.from_numpy(pic)
        img = img.unsqueeze(1)
        default_float_dtype = torch.get_default_dtype()
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype).div(255)
        else:
            return img


class ZScoreNormalizationPerSample:
    """Apply Z-score normalization to a tensor based on its own mean and std."""

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        mean = tensor.mean()
        std = tensor.std()
        std_adj = std.clone()
        if std_adj == 0:
            std_adj = 1.0
        return (tensor - mean) / std_adj


