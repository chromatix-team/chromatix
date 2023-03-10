import jax.numpy as jnp
from ..field import Field
from einops import rearrange
from ..utils import center_pad, center_crop
from ..ops.fft import fftshift, fft, ifft, ifftshift
from typing import Literal, Optional, Tuple, Union
from chex import Array
import numpy as np

__all__ = [
    "transform_propagate",
    "transfer_propagate",
    "exact_propagate",
    "kernel_propagate",
    "compute_transfer_propagator",
    "compute_exact_propagator",
    "compute_padding_transform",
    "compute_padding_transfer",
    "compute_padding_exact",
]


def transform_propagate(
    field: Field,
    z: Union[float, Array],
    n: float,
    N_pad: int,
    cval: float = 0,
    loop_axis: Optional[int] = None,
) -> Field:
    """
    Fresnel propagate ``field`` for a distance ``z`` using transform method.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
        propagation FFT
    """
    z = jnp.atleast_1d(z)
    z = rearrange(z, "z ->" + " 1" * (field.rank - 4) + " z 1 1 1")
    # Fourier normalization factor
    L = jnp.sqrt(field.spectrum * z / n)  # lengthscale L
    norm = (field.dx / L) ** 2
    # Calculating input phase change
    input_phase = jnp.pi * field.l2_sq_grid / L**2
    # Calculating new scaled output coordinates
    du = L**2 / ((field.spatial_shape[0] + N_pad) * field.dx)
    # Calculating output phase
    output_grid = field.l2_sq_grid * (du / field.dx) ** 2
    output_phase = jnp.pi * output_grid / L**2
    # Determining new field
    u = field.u * jnp.exp(1j * input_phase)
    pad = [0] * len(field.shape)
    for d in field.spatial_dims:
        pad[d] = N_pad // 2
    u = center_pad(u, pad, cval=cval)
    u = fftshift(fft(ifftshift(u, axes=field.spatial_dims), axes=field.spatial_dims, loop_axis=loop_axis), axes=(field.spatial_dims))
    u = center_crop(u, pad)
    # Final normalization and phase
    u *= norm * jnp.exp(1j * output_phase)
    return field.replace(u=u, dx=du)


def transfer_propagate(
    field: Field,
    z: Union[float, Array],
    n: float,
    N_pad: int,
    cval: float = 0,
    kykx: Array = jnp.zeros((2,)),
    loop_axis: Optional[int] = None,
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Fresnel propagate ``field`` for a distance ``z`` using transfer method.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
        propagation FFT (NOTE: should not be a Jax array, otherwise a
        ConcretizationError will arise when traced!).
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    propagator = compute_transfer_propagator(
        field.shape, field.dx, field.spectrum, z, n, N_pad, kykx=kykx, spatial_dims=field.spatial_dims
    )
    return kernel_propagate(field, propagator, N_pad, cval, loop_axis, mode)


def exact_propagate(
    field: Field,
    z: Union[float, Array],
    n: float,
    N_pad: int,
    cval: float = 0,
    kykx: Array = jnp.zeros((2,)),
    loop_axis: Optional[int] = None,
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Propagate ``field`` for a distance ``z`` using exact transfer method.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a Jax array, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            compute the padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    propagator = compute_exact_propagator(
        field.u.shape, field.dx, field.spectrum, z, n, N_pad, kykx=kykx, spatial_dims=field.spatial_dims
    )
    return kernel_propagate(field, propagator, N_pad, cval, loop_axis, mode)


def kernel_propagate(
    field: Field,
    propagator: Array,
    N_pad: int,
    cval: float = 0,
    loop_axis: Optional[int] = None,
    mode: Literal["full", "same"] = "full",
) -> Field:
    # Propagating field
    pad = [0] * len(field.shape)
    for d in field.spatial_dims:
        pad[d] = N_pad // 2
    u = center_pad(field.u, pad, cval=cval)
    u = ifft(fft(u, axes=field.spatial_dims, loop_axis=loop_axis) * propagator, axes=field.spatial_dims, loop_axis=loop_axis)
    # Cropping output field
    if mode == "full":
        field = field.replace(u=u)
    elif mode == "same":
        u = center_crop(u, pad)
        field = field.replace(u=u)
    else:
        raise NotImplementedError('Only "full" and "same" are supported.')
    return field


def _frequency_grid(
    shape: Tuple[int, ...],
    dx: Union[float, Array],
    N_pad: int,
    spatial_dims: Tuple[int, int]
) -> Tuple[Array, Array]:
    rank = len(shape)
    dx = jnp.atleast_1d(dx)
    # TODO(dd): This calculation could probably go into Field
    f = []
    for d in range(dx.size):
        f.append(jnp.fft.fftfreq(shape[spatial_dims[0]] + N_pad, d=dx[..., d].squeeze()))
    f = jnp.stack(f, axis=-1)
    fx = rearrange(f, "h c -> " + "1 " * (rank - 3) + "h 1 c")
    fy = rearrange(f, "w c -> " + "1 " * (rank - 3) + "1 w c")
    return fx, fy


def compute_transfer_propagator(
    shape: Tuple[int, ...],
    dx: Union[float, Array],
    spectrum: Union[float, Array],
    z: Union[float, Array],
    n: float,
    N_pad: int,
    kykx: Array = jnp.zeros((2,)),
    spatial_dims: Tuple[int, int] = (1, 2)
):
    """Compute propagation kernel for Fresnel propagation.

    Returns an array that can be multiplied with the Fourier transform of the
    incoming Field, as performed by kernel_propagate.

    Args:
        shape: Shape of the propagator.
        dx: The spacing of the incoming ``Field``.
        spectrum: Spectrum of the incoming ``Field``.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a Jax array, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            compute the padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
    """
    rank = len(shape)
    z = jnp.atleast_1d(z)
    # z = rearrange(z, "z -> z" + " 1" * (rank - 1))
    z = rearrange(z, "z ->" + " 1" * (rank - 4) + " z 1 1 1")
    L = jnp.sqrt(jnp.complex64(spectrum * z / n))  # lengthscale L
    fx, fy = _frequency_grid(shape, dx, N_pad, spatial_dims)
    # dx = jnp.atleast_1d(dx)
    # f = []
    # for d in range(dx.size):
    #     f.append(jnp.fft.fftfreq(shape[1] + N_pad, d=dx[..., d].squeeze()))
    # f = jnp.stack(f, axis=-1)
    # fx, fy = rearrange(f, "h c -> 1 h 1 c"), rearrange(f, "w c -> 1 1 w c")
    phase = -jnp.pi * L**2 * ((fx - kykx[1]) ** 2 + (fy - kykx[0]) ** 2)
    return jnp.exp(1j * phase)


def compute_exact_propagator(
    shape: Tuple[int, ...],
    dx: Union[float, Array],
    spectrum: Union[float, Array],
    z: Union[float, Array],
    n: float,
    N_pad: int,
    kykx: Array = jnp.zeros((2,)),
    spatial_dims: Tuple[int, int] = (1, 2)
):
    """Compute propagation kernel for propagation with no Fresnel approximation.

    Returns an array that can be multiplied with the Fourier transform of the
    incoming Field, as performed by kernel_propagate.

    Args:
        shape: Shape of the propagator.
        dx: The spacing of the incoming ``Field``.
        spectrum: Spectrum of the incoming ``Field``.
        z: Distance(s) to propagate, either a float or an array of shape (Z 1
            1 1).
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a Jax array, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            compute the padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
    """
    rank = len(shape)
    z = jnp.atleast_1d(z)
    z = rearrange(z, "z ->" + " 1" * (rank - 4) + " z 1 1 1")
    fx, fy = _frequency_grid(shape, dx, N_pad, spatial_dims)
    # dx = jnp.atleast_1d(dx)
    # f = []
    # if isinstance(dx, float):
    #     dx = rearrange(jnp.atleast_1d(dx), "c -> 1 1 1 c")
    # for d in range(dx.size):
    #     f.append(jnp.fft.fftfreq(shape[1] + N_pad, d=dx[..., d].squeeze()))
    # f = jnp.stack(f, axis=-1)
    # fx, fy = rearrange(f, "h c -> 1 h 1 c"), rearrange(f, "w c -> 1 1 w c")
    kernel = 1 - (spectrum / n) ** 2 * ((fx - kykx[1]) ** 2 + (fy - kykx[0]) ** 2)
    kernel = jnp.maximum(kernel, 0.0)  # removing evanescent waves
    phase = 2 * jnp.pi * (z * n / spectrum) * jnp.sqrt(kernel)
    return jnp.exp(1j * phase)


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
