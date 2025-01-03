import random
import math
from collections.abc import Sequence
import torch
from torch import nn
import torch.nn.functional as F
from utils import GaussianFilter


class CenterCrop3D(nn.Module):
    """Crop the input 3D tensor at the center.

    Args:
        size (int or sequence): Desired output size of the crop (d, h, w).
            If size is an int, a cubic output of (size, size, size) will be produced.
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
        """
        Args:
            img (Tensor): Input 3D image tensor.
            output_size (tuple): Desired output size (d, h, w).

        Returns:
            tuple: Parameters (i, j, k, d, h, w) for the crop.
        """
        depth, height, width = img.shape[-3:]
        d, h, w = output_size

        if depth < d or height < h or width < w:
            raise ValueError(
                f"Requested crop size {output_size} is larger than input image size "
                f"({depth}, {height}, {width})."
            )

        i = (depth - d) // 2
        j = (height - h) // 2
        k = (width - w) // 2

        return i, j, k, d, h, w

    def forward(self, img):
        """
        Args:
            img (Tensor): 3D image to be center-cropped.

        Returns:
            Tensor: Center-cropped 3D image.
        """
        i, j, k, d, h, w = self.get_params(img, self.size)
        return img[..., i:i + d, j:j + h, k:k + w]


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
        """Get parameters for a random sized crop without using a for-loop.

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

        # Generate N candidate crops in parallel
        N = 10  # Number of candidates
        target_volumes = area * torch.empty(N).uniform_(scale[0], scale[1])
        aspects = torch.exp(torch.empty(N).uniform_(log_ratio[0], log_ratio[1]))

        # Compute dimensions of the crop
        ds = torch.round((target_volumes * aspects) ** (1/3)).int()
        hs = torch.round((target_volumes / aspects) ** (1/3)).int()
        ws = torch.round((target_volumes / aspects) ** (1/3)).int()

        # Filter out invalid dimensions
        valid = (ds <= depth) & (hs <= height) & (ws <= width)
        valid_indices = valid.nonzero(as_tuple=False).view(-1)

        if valid_indices.numel() > 0:
            # Randomly select one of the valid candidates
            idx = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
            d = ds[idx].item()
            h = hs[idx].item()
            w = ws[idx].item()

            # Randomly select the crop position
            i = torch.randint(0, depth - d + 1, (1,)).item()
            j = torch.randint(0, height - h + 1, (1,)).item()
            k = torch.randint(0, width - w + 1, (1,)).item()
            return i, j, k, d, h, w

        # Fallback to center crop if no valid candidates
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


class RandGaussianSmooth(nn.Module):
    """
    Randomly apply Gaussian smoothing using MONAI's GaussianFilter.

    Args:
        spatial_dims: number of spatial dimensions (e.g., 3 for a 3D volume).
        sigma_range (tuple): range of sigma values from which a random sigma is drawn.
        prob (float): probability of applying the smoothing.
        truncated (float): how many std devs for the kernel size.
        approx (str): approximation method for Gaussian kernel, e.g. 'erf', 'sampled', 'scalespace'.
    """
    def __init__(self, spatial_dims=3, sigma_range=(0.5, 1.5), prob=0.5, truncated=4.0, approx="erf"):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.sigma_range = sigma_range
        self.prob = prob
        self.truncated = truncated
        self.approx = approx

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Probability check
        if random.random() > self.prob:
            return img

        # Choose a random sigma (one for each dimension if you like, or the same for all)
        # Here we pick one sigma and use it for all dimensions for simplicity.
        sigma_val = random.uniform(*self.sigma_range)
        sigma = [sigma_val] * self.spatial_dims

        # Ensure img has shape: [N, C, D, H, W] if 3D
        if img.dim() == 3:  # (D, H, W)
            img = img.unsqueeze(0).unsqueeze(0)  # (N=1, C=1, D, H, W)
        elif img.dim() == 4: # (C, D, H, W)
            img = img.unsqueeze(0)  # (N=1, C, D, H, W)

        # Create GaussianFilter with the chosen sigma
        gauss_filter = GaussianFilter(
            spatial_dims=self.spatial_dims,
            sigma=sigma,
            truncated=self.truncated,
            approx=self.approx,
            requires_grad=False
        )

        # Apply filter
        img = gauss_filter(img)

        # Remove added dimensions if necessary
        if img.shape[0] == 1 and img.dim() > 4:
            img = img.squeeze(0)

        return img


class RandomGamma3D(nn.Module):
    """
    Randomly apply gamma correction to the volume based on a log-gamma range, similar
    to the logic in TorchIO's RandomGamma.

    Args:
        log_gamma (float or tuple of floats): If a tuple (a, b), we sample a value
            log_gamma_val ~ Uniform(a, b) and set gamma = exp(log_gamma_val).
            If a single float d is provided, we use (-d, d).
        prob (float): Probability of applying gamma correction.
    """
    def __init__(self, log_gamma=(-0.3, 0.3), prob=0.5):
        super().__init__()
        if isinstance(log_gamma, (int, float)):
            # If a single value is given, interpret as (-d, d)
            log_gamma = (-log_gamma, log_gamma)
        self.log_gamma = log_gamma
        self.prob = prob

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return img

        # Sample a log-gamma value and exponentiate to get gamma
        log_gamma_val = random.uniform(self.log_gamma[0], self.log_gamma[1])
        gamma = math.exp(log_gamma_val)

        # Apply gamma correction
        # If there are negative values, use: sign(x)*|x|^gamma
        # Otherwise, use x^gamma directly.
        if (img < 0).any():
            return img.sign() * (img.abs() ** gamma)
        else:
            return img ** gamma