from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from collections.abc import Sequence

from monai.utils import issequenceiterable
from monai.networks.layers.convutils import gaussian_1d


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _separable_filtering_conv(
    input_: torch.Tensor,
    kernels: list[torch.Tensor],
    pad_mode: str,
    d: int,
    spatial_dims: int,
    paddings: list[int],
    num_channels: int,
) -> torch.Tensor:
    if d < 0:
        return input_

    s = [1] * len(input_.shape)
    s[d + 2] = -1
    _kernel = kernels[d].reshape(s)

    # if filter kernel is unity, don't convolve
    if _kernel.numel() == 1 and _kernel[0] == 1:
        return _separable_filtering_conv(input_, kernels, pad_mode, d - 1, spatial_dims, paddings, num_channels)

    _kernel = _kernel.repeat([num_channels, 1] + [1] * spatial_dims)
    _padding = [0] * spatial_dims
    _padding[d] = paddings[d]
    conv_type = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]

    # translate padding for input to torch.nn.functional.pad
    _reversed_padding_repeated_twice: list[list[int]] = [[p, p] for p in reversed(_padding)]
    _sum_reversed_padding_repeated_twice: list[int] = sum(_reversed_padding_repeated_twice, [])
    padded_input = F.pad(input_, _sum_reversed_padding_repeated_twice, mode=pad_mode)

    return conv_type(
        input=_separable_filtering_conv(padded_input, kernels, pad_mode, d - 1, spatial_dims, paddings, num_channels),
        weight=_kernel,
        groups=num_channels,
    )


def separable_filtering(x: torch.Tensor, kernels: list[torch.Tensor], mode: str = "zeros") -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")

    spatial_dims = len(x.shape) - 2
    if isinstance(kernels, torch.Tensor):
        kernels = [kernels] * spatial_dims
    _kernels = [s.to(x) for s in kernels]
    _paddings = [(k.shape[0] - 1) // 2 for k in _kernels]
    n_chs = x.shape[1]
    pad_mode = "constant" if mode == "zeros" else mode

    return _separable_filtering_conv(x, _kernels, pad_mode, spatial_dims - 1, spatial_dims, _paddings, n_chs)


class GaussianFilter(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        sigma: Sequence[float] | float | Sequence[torch.Tensor] | torch.Tensor,
        truncated: float = 4.0,
        approx: str = "erf",
        requires_grad: bool = False,
    ) -> None:
        if issequenceiterable(sigma):
            if len(sigma) != spatial_dims:  # type: ignore
                raise ValueError
        else:
            sigma = [deepcopy(sigma) for _ in range(spatial_dims)]  # type: ignore
        super().__init__()
        self.sigma = [
            torch.nn.Parameter(
                torch.as_tensor(s, dtype=torch.float, device=s.device if isinstance(s, torch.Tensor) else None),
                requires_grad=requires_grad,
            )
            for s in sigma  # type: ignore
        ]
        self.truncated = truncated
        self.approx = approx
        for idx, param in enumerate(self.sigma):
            self.register_parameter(f"kernel_sigma_{idx}", param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: in shape [Batch, chns, H, W, D].
        """
        _kernel = [gaussian_1d(s, truncated=self.truncated, approx=self.approx) for s in self.sigma]
        return separable_filtering(x=x, kernels=_kernel)

