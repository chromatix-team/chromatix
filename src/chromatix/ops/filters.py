from typing import Optional, Sequence, Tuple
from .ops import fourier_convolution
from chex import Array
from chromatix.utils import gaussian_kernel
from chromatix.utils import _broadcast_2d_to_spatial

__all__ = ["high_pass_filter", "gaussian_filter"]


def high_pass_filter(
    data: Array,
    sigma: Sequence[float],
    axes: Tuple[int, int] = (1, 2),
    kernel_shape: Optional[Sequence[int]] = None,
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
    assert len(axes) == len(
        sigma
    ), "Must specify same number of axes to convolve as elements in sigma"
    low_pass_kernel = gaussian_kernel(sigma, shape=kernel_shape)
    # NOTE(gj): 1e-3 effectively gives delta kernel
    delta_kernel = gaussian_kernel((1e-3,) * len(sigma), shape=low_pass_kernel.shape)
    kernel = delta_kernel - low_pass_kernel
    kernel = _broadcast_2d_to_spatial(kernel, data.ndim)
    return fourier_convolution(data, kernel, axes=axes)


def gaussian_filter(
    data: Array,
    sigma: Sequence[float],
    axes: Tuple[int, int] = (1, 2),
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
    assert len(axes) == len(
        sigma
    ), "Must specify same number of axes to convolve as elements in sigma"
    kernel = gaussian_kernel(sigma, shape=kernel_shape)
    kernel = _broadcast_2d_to_spatial(kernel, data.ndim)
    return fourier_convolution(data, kernel, axes=axes)
