from typing import Optional, Sequence, Tuple
from .ops import fourier_convolution
from chex import Array
import jax.numpy as jnp

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
    return fourier_convolution(data, kernel, axes=axes)


def gaussian_kernel(
    sigma: Sequence[float], truncate: float = 4.0, shape: Optional[Sequence[int]] = None
) -> Array:
    """
    Creates ND Gaussian kernel of given ``sigma``.

    If ``shape`` is not provided, then the shape of the kernel is automatically
    calculated using the given truncation (the same truncation for each
    dimension) and ``sigma``. The number of dimensions is determined by the
    length of ``sigma``, which should be a 1D array.

    If ``shape`` is provided, then ``truncate`` is ignored and the result will
    have the provided ``shape``. The provided ``shape`` must be odd in all
    dimensions to ensure that there is a center pixel.

    Args:
        sigma: A 1D array whose length is the number of dimensions specifying
            the standard deviation of the Gaussian distribution in each
            dimension.
        truncate: If ``shape`` is not provided, then this float is the number
            of standard deviations for which to calculate the Gaussian. This is
            then used to determine the shape of the kernel in each dimension.
        shape: If provided, determines the ``shape`` of the kernel. This will
            cause ``truncate`` to be ignored.

    Returns:
        The ND Gaussian kernel.
    """
    _sigma = jnp.atleast_1d(jnp.array(sigma))
    if shape is not None:
        _shape = jnp.atleast_1d(jnp.array(shape))
        assert jnp.all(_shape % 2 != 0), "Shape must be odd in all dimensions"
        radius = ((_shape - 1) / 2).astype(jnp.int16)
    else:
        radius = (truncate * _sigma + 0.5).astype(jnp.int16)

    x = jnp.mgrid[tuple(slice(-r, r + 1) for r in radius)]
    phi = jnp.exp(-0.5 * jnp.sum((x.T / _sigma) ** 2, axis=-1))  # type: ignore
    return phi / phi.sum()
