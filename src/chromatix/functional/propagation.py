import jax
import jax.numpy as jnp
from ..field import Field
from einops import rearrange
from ..utils import center_pad, center_crop
from ..ops.fft import fftshift, fft, ifft, ifftshift
from typing import Optional, Tuple
from chex import Array
import numpy as np
from functools import partial

__all__ = [
    "transform_propagate",
    "transfer_propagate",
    "exact_propagate",
    "calculate_padding_transform",
    "calculate_padding_transfer",
    "calculate_padding_exact",
]


def transform_propagate(
    field: Field, z: float, n: float, *, N_pad: int, loop_axis: Optional[int] = None
) -> Field:
    """
    Fresnel propagate ``field`` for a distance ``z`` using transform method.

    Args:
        field: ``Field`` to be propagated.
        z: A float that defines the distance to propagate.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
        propagation FFT
    """

    # Fourier normalization factor
    # assert N_pad % 2 == 0, "Padding should be even."
    L = jnp.sqrt(field.spectrum * z / n)  # lengthscale L
    norm = (field.dx / L) ** 2

    # Calculating input phase change
    input_phase = jnp.pi * field.l2_sq_grid / L**2

    # Calculating new scaled output coordinates
    du = L**2 / ((field.shape[1] + N_pad) * field.dx)

    # Calculating output phase
    output_grid = field.l2_sq_grid * (du / field.dx) ** 2
    output_phase = jnp.pi * output_grid / L**2

    # Determining new field
    u = field.u * jnp.exp(1j * input_phase)
    u = center_pad(u, [0, int(N_pad / 2), int(N_pad / 2), 0])
    u = fftshift(fft(ifftshift(u), loop_axis))
    u = center_crop(u, [0, int(N_pad / 2), int(N_pad / 2), 0])

    # Final normalization and phase
    u *= norm * jnp.exp(1j * output_phase)

    return field.replace(u=u, dx=du)


def transfer_propagate(
    field: Field,
    z: float,
    n: float,
    *,
    N_pad: int,
    loop_axis: Optional[int] = None,
    mode: str = "full",
) -> Field:
    """
    Fresnel propagate ``field`` for a distance ``z`` using transfer method.

    Args:
        field: ``Field`` to be propagated.
        z: A float that defines the distance to propagate.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
        propagation FFT (NOTE: should not be a Jax array, otherwise a
        ConcretizationError will arise when traced!).
    """

    # assert N_pad % 2 == 0, "Padding should be even."
    # Calculating propagator
    L = jnp.sqrt(jnp.complex64(field.spectrum * z / n))  # lengthscale L
    # TODO(dd): This calculation could probably go into Field
    f = []
    for d in range(field.dx.size):
        f.append(jnp.fft.fftfreq(field.shape[1] + N_pad, d=field.dx[..., d].squeeze()))
    f = jnp.stack(f, axis=-1)
    fx, fy = rearrange(f, "h c -> 1 h 1 c"), rearrange(f, "w c -> 1 1 w c")
    phase = -jnp.pi * L**2 * (fx**2 + fy**2)

    # Propagating field
    u = center_pad(field.u, [0, int(N_pad / 2), int(N_pad / 2), 0])
    u = ifft(fft(u, loop_axis) * jnp.exp(1j * phase), loop_axis)

    # Cropping output field
    if mode == "full":
        field = field.replace(u=u)
    elif mode == "same":
        u = center_crop(u, [0, int(N_pad / 2), int(N_pad / 2), 0])
        field = field.replace(u=u)
    else:
        raise NotImplementedError('Only "full" and "same" are supported.')

    return field


def exact_propagate(
    field: Field,
    z: float,
    n: float,
    *,
    N_pad: int,
    cval: float = 0,
    kykx: Array = jnp.zeros((2,)),
    loop_axis: Optional[int] = None,
    mode: str = "full",
) -> Field:
    """
    Propagate ``field`` for a distance ``z`` using exact transfer method.

    Args:
        field: ``Field`` to be propagated.
        z: A float that defines the distance to propagate.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a Jax array, otherwise a
            ConcretizationError will arise when traced!). Use padding calculator
            utilities from ``chromatix.functional.propagation`` to calculate the
            padding.
        cval: The value for the padding, 0 by default
        kykx: If provided, defines the orientation of the propagation. Should be an
            array of shape `[2,]` in the format [ky, kx].
    """
    # Calculating propagator
    f = []
    for d in range(field.dx.size):
        f.append(jnp.fft.fftfreq(field.shape[1] + N_pad, d=field.dx[..., d].squeeze()))
    f = jnp.stack(f, axis=-1)
    fx, fy = rearrange(f, "h c -> 1 h 1 c"), rearrange(f, "w c -> 1 1 w c")
    kernel = 1 - (field.spectrum / n) ** 2 * ((fx - kykx[1]) ** 2 + (fy - kykx[0]) ** 2)
    kernel = jnp.maximum(kernel, 0.0)  # removing evanescent waves
    phase = 2 * jnp.pi * (z * n / field.spectrum) * jnp.sqrt(kernel)

    # Propagating field
    u = center_pad(field.u, [0, int(N_pad / 2), int(N_pad / 2), 0], cval=cval)
    u = ifft(fft(u, loop_axis) * jnp.exp(1j * phase), loop_axis)

    # Cropping output field
    if mode == "full":
        field = field.replace(u=u)
    elif mode == "same":
        u = center_crop(u, [0, int(N_pad / 2), int(N_pad / 2), 0])
        field = field.replace(u=u)
    else:
        raise NotImplementedError('Only "full" and "same" are supported.')

    return field


def calculate_exact_kernel(
    shape: Tuple,
    dx: Array,
    spectrum,
    z: float,
    n: float,
    N_pad: int,
    kykx: Array = jnp.zeros((2,)),
):
    f = []
    dx = rearrange(jnp.atleast_1d(dx), "c -> 1 1 1 c")
    for d in range(dx.size):
        f.append(jnp.fft.fftfreq(shape[1] + N_pad, d=dx[..., d].squeeze()))
    f = jnp.stack(f, axis=-1)
    fx, fy = rearrange(f, "h c -> 1 h 1 c"), rearrange(f, "w c -> 1 1 w c")
    kernel = 1 - (spectrum / n) ** 2 * ((fx - kykx[1]) ** 2 + (fy - kykx[0]) ** 2)
    kernel = jnp.maximum(kernel, 0.0)  # removing evanescent waves
    phase = 2 * jnp.pi * (z * n / spectrum) * jnp.sqrt(kernel)

    return jnp.exp(1j * phase)


def calculate_padding_transform(
    height: int, spectrum: float, dx: float, z: float
) -> int:
    """
    Automatically calculate the padding required for transform propagation.

    #TODO: works only for square fields
    Args:
        height: Height of the field
        spectrum: spectrum of the field
        dx: spacing of the field
        z: A float that defines the distance to propagate.
    """
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (spectrum * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2

    N = (np.ceil((Q * M) / 2) * 2).astype(int)
    N_pad = ((N - M)).astype(int)

    return N_pad


def calculate_padding_transfer(
    height: int, spectrum: float, dx: float, z: float
) -> int:
    """
    Automatically calculate the padding required for transfer propagation.

    Args:
        height: Height of the field
        spectrum: spectrum of the field
        dx: spacing of the field
        z: A float that defines the distance to propagate.
    """
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (spectrum * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2

    N = (jnp.ceil((Q * M) / 2) * 2).astype(int)
    N_pad = (N - M).astype(int)

    return N_pad


def calculate_padding_exact(height: int, spectrum: float, dx: float, z: float) -> int:
    """
    Automatically calculate the padding required for exact propagation.

    Args:
        height: Height of the field
        spectrum: spectrum of the field
        dx: spacing of the field
        z: A float that defines the distance to propagate.
    """
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
