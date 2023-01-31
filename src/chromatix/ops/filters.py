from typing import Sequence, Optional
from ..utils.utils import gaussian_kernel
from . import fourier_convolution
from einops import repeat
from jax import vmap
from chex import Array


def high_pass_filter(
    data: Array, sigma: float, kernel_shape: Optional[Sequence[int]] = None
) -> Array:
    low_pass_kernel = gaussian_kernel((sigma, sigma), shape=kernel_shape)
    # 1e-3 effetively gives delta kernel
    delta_kernel = gaussian_kernel((1e-3, 1e-3), shape=low_pass_kernel.shape)
    kernel = repeat(delta_kernel - low_pass_kernel, "h w -> n h w 1", n=data.shape[0])
    return vmap(fourier_convolution)(data, kernel)


def gaussian_filter(
    data: Array,
    sigma: Sequence[float],
    kernel_shape: Optional[Sequence[int]] = None,
) -> Array:
    kernel = gaussian_kernel(sigma, shape=kernel_shape)
    return fourier_convolution(data, kernel)
