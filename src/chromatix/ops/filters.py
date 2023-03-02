from typing import Sequence, Optional
from ..utils.utils import gaussian_kernel
from . import fourier_convolution
from einops import repeat
from jax import vmap
from chex import Array


def high_pass_filter(
    data: Array, sigma: float, kernel_shape: Optional[Sequence[int]] = None
) -> Array:
    """
    Performs a high pass filter on ``data``.

    The high pass filter is constructed as the difference between a
    delta kernel and a Gaussian kernel with standard deviation ``sigma``.

    Args:
        data: The input to be high pass filtered.
        sigma: The standard deviation of the Gaussian kernel, which sets the
            low pass filter. The result of this low pass will be subtracted
            from the input.
        kernel_shape: The shape of the kernel. If not provided, the shape will
            be determined by the required shape of a Gaussian kernel truncated
            to ``4.0 * sigma``.

    Returns:
        The high pass filtered array.
    """
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
    """
    Performs a Gaussian filter on ``data``.

    Args:
        data: The input to be Gaussian filtered.
        sigma: The standard deviation of the Gaussian kernel.
        kernel_shape: The shape of the kernel. If not provided, the shape will
            be determined by the required shape of a Gaussian kernel truncated
            to ``4.0 * sigma``.

    Returns:
        The Gaussian filtered array.
    """
    kernel = gaussian_kernel(sigma, shape=kernel_shape)
    return fourier_convolution(data, kernel)
