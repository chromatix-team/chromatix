import numpy as np
import jax.numpy as jnp
from chex import Array
from typing import Literal, Tuple, Union
from ..field import Field
from ..utils import _broadcast_1d_to_innermost_batch, _broadcast_1d_to_grid, l2_sq_norm
from ..ops.fft import fft, ifft, optical_fft
from ..ops.field import pad, crop

__all__ = [
    "transform_propagate",
    "transfer_propagate",
    "exact_propagate",
    "asm_propagate",
    "kernel_propagate",
    "compute_transfer_propagator",
    "compute_exact_propagator",
    "compute_asm_propagator",
    "compute_padding_transform",
    "compute_padding_transfer",
    "compute_padding_exact",
]


def transform_propagate(
    field: Field,
    z: Union[float, Array],
    n: float,
    N_pad: Union[int, Tuple[int, int]],
    cval: float = 0,
) -> Field:
    """
    Fresnel propagate ``field`` for a distance ``z`` using transform method.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a Jax array, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            compute the padding.
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
    """
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    field = pad(field, N_pad, cval=cval)
    # Fourier normalization factor
    L = jnp.sqrt(jnp.complex64(field.spectrum * z / n))  # lengthscale L
    # Calculating input phase change
    input_phase = jnp.pi * l2_sq_norm(field.grid) / jnp.abs(L) ** 2
    # New field is optical_fft minus -1j factor
    field = 1j * optical_fft(field * jnp.exp(1j * input_phase), z, n)
    output_phase = jnp.pi * l2_sq_norm(field.grid) / jnp.abs(L) ** 2
    field = field * jnp.exp(1j * output_phase)
    return crop(field, N_pad)


def transfer_propagate(
    field: Field,
    z: Union[float, Array],
    n: float,
    N_pad: int,
    cval: float = 0,
    kykx: Array = jnp.zeros((2,)),
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Fresnel propagate ``field`` for a distance ``z`` using transfer method.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a Jax array, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            compute the padding.
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    field = pad(field, N_pad, cval=cval)
    propagator = compute_transfer_propagator(field, z, n, kykx)
    field = kernel_propagate(field, propagator)
    if mode == "same":
        field = crop(field, N_pad)
    return field


def exact_propagate(
    field: Field,
    z: Union[float, Array],
    n: float,
    N_pad: int,
    cval: float = 0,
    kykx: Array = jnp.zeros((2,)),
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Propagate ``field`` for a distance ``z`` using exact transfer method.

    This method removes evanescent waves.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a Jax array, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            compute the padding.
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    field = pad(field, N_pad, cval=cval)
    propagator = compute_exact_propagator(field, z, n, kykx)
    field = kernel_propagate(field, propagator)
    if mode == "same":
        field = crop(field, N_pad)
    return field


def asm_propagate(
    field: Field,
    z: Union[float, Array],
    n: float,
    N_pad: int,
    cval: float = 0,
    kykx: Array = jnp.zeros((2,)),
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Propagate ``field`` for a distance ``z`` using angular spectrum method.

    This method does not remove evanescent waves.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a Jax array, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            compute the padding.
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    field = pad(field, N_pad, cval=cval)
    propagator = compute_asm_propagator(field, z, n, kykx)
    field = kernel_propagate(field, propagator)
    if mode == "same":
        field = crop(field, N_pad)
    return field


def kernel_propagate(field: Field, propagator: Array) -> Field:
    """
    Propagate an incoming ``Field`` by the given propagation kernel
    (``propagator``). This amounts to performing a Fourier convolution of the
    ``field`` and the ``propagator``.
    """
    axes = field.spatial_dims
    u = ifft(fft(field.u, axes=axes) * propagator, axes=axes)
    return field.replace(u=u)


def compute_transfer_propagator(
    field: Field,
    z: Union[float, Array],
    n: float,
    kykx: Array = jnp.zeros((2,)),
) -> Array:
    """
    Compute propagation kernel for Fresnel propagation.
    Returns an array that can be multiplied with the Fourier transform of the
    incoming Field, as performed by kernel_propagate.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
    """
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    L = jnp.sqrt(jnp.complex64(field.spectrum * z / n))  # lengthscale L
    phase = -jnp.pi * jnp.abs(L) ** 2 * l2_sq_norm(field.k_grid - kykx)
    return jnp.fft.ifftshift(jnp.exp(1j * phase), axes=field.spatial_dims)


def compute_exact_propagator(
    field: Field,
    z: Union[float, Array],
    n: float,
    kykx: Array = jnp.zeros((2,)),
) -> Array:
    """
    Compute propagation kernel for propagation with no Fresnel approximation.

    This version of the propagation kernel removes evanescent waves. Returns
    an array that can be multiplied with the Fourier transform of the incoming
    Field, as performed by kernel_propagate.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or an array of shape (Z 1
            1 1).
        n: A float that defines the refractive index of the medium.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
    """
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    kernel = 1 - (field.spectrum / n) ** 2 * l2_sq_norm(field.k_grid - kykx)
    kernel = jnp.maximum(kernel, 0.0)  # removing evanescent waves
    phase = 2 * jnp.pi * (z * n / field.spectrum) * jnp.sqrt(kernel)
    return jnp.fft.ifftshift(jnp.exp(1j * phase), axes=field.spatial_dims)


def compute_asm_propagator(
    field: Field,
    z: Union[float, Array],
    n: float,
    kykx: Array = jnp.zeros((2,)),
) -> Array:
    """
    Compute propagation kernel for propagation with no Fresnel approximation.

    This version of the propagation kernel does not remove evanescent waves,
    as per the definition of the angular spectrum method. Returns an array
    that can be multiplied with the Fourier transform of the incoming Field, as
    performed by kernel_propagate.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or an array of shape (Z 1
            1 1).
        n: A float that defines the refractive index of the medium.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
    """
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    kernel = 1 - (field.spectrum / n) ** 2 * l2_sq_norm(field.k_grid - kykx)
    delay = jnp.sqrt(jnp.abs(kernel))
    delay = jnp.where(kernel >= 0, delay, 1j * delay)  # keep evanescent modes
    phase = 2 * jnp.pi * (z * n / field.spectrum) * delay
    return jnp.fft.ifftshift(jnp.exp(1j * phase), axes=field.spatial_dims)


def compute_padding_transform(height: int, spectrum: float, dx: float, z: float) -> int:
    """
    Automatically compute the padding required for transform propagation.

    Args:
        height: Height of the field
        spectrum: spectrum of the field
        dx: spacing of the field
        z: A float that defines the distance to propagate.
    """
    # TODO: works only for square fields
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (spectrum * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2
    N = (np.ceil((Q * M) / 2) * 2).astype(int)
    N_pad = ((N - M)).astype(int)
    return N_pad


def compute_padding_transfer(height: int, spectrum: float, dx: float, z: float) -> int:
    """
    Automatically compute the padding required for transfer propagation.

    Args:
        height: Height of the field
        spectrum: spectrum of the field
        dx: spacing of the field
        z: A float that defines the distance to propagate.
    """
    # TODO: works only for square fields
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (spectrum * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2
    N = (jnp.ceil((Q * M) / 2) * 2).astype(int)
    N_pad = (N - M).astype(int)
    return N_pad


def compute_padding_exact(height: int, spectrum: float, dx: float, z: float) -> int:
    """
    Automatically compute the padding required for exact propagation.

    Args:
        height: Height of the field
        spectrum: spectrum of the field
        dx: spacing of the field
        z: A float that defines the distance to propagate.
    """
    # TODO: works only for square fields
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (spectrum * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2
    scale = np.max((spectrum / (2 * dx)))
    # assert scale < 1, "Can't do exact transfer when field.dx < lambda / 2"
    Q = Q / np.sqrt(1 - scale**2)  # minimum pad ratio for exact transfer
    N = (np.ceil((Q * M) / 2) * 2).astype(int)
    N_pad = (N - M).astype(int)
    return N_pad
