import jax
import jax.numpy as jnp
from ..field import Field
from einops import rearrange
from ..utils import center_pad, center_crop
from ..ops.fft import fftshift, fft, ifft
from typing import Optional

__all__ = ["propagate", "transform_propagate", "transfer_propagate", "exact_propagate"]


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
    u = center_pad(u, [0, N_pad // 2, N_pad // 2, 0])
    u = fftshift(fft(u, loop_axis))
    u = center_crop(u, [0, N_pad // 2, N_pad // 2, 0])

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
    if field.u.ndim > 4:
        for d in range(field.dx.size):
            f.append(
                jnp.fft.fftfreq(field.shape[2] + N_pad, d=field.dx[..., d].squeeze())
            )
    else:
        for d in range(field.dx.size):
            f.append(
                jnp.fft.fftfreq(field.shape[1] + N_pad, d=field.dx[..., d].squeeze())
            )
    f = jnp.stack(f, axis=-1)

    if field.u.ndim > 4:
        fx, fy = rearrange(f, "h c -> 1 1 h 1 c"), rearrange(f, "w c -> 1 1 1 w c")
        u = center_pad(field.u, [0, 0, int(N_pad / 2), int(N_pad / 2), 0])
        print("fx shape is:", fx.shape)
        phase = -jnp.pi * L**2 * (fx**2 + fy**2)
    else:
        fx, fy = rearrange(f, "h c -> 1 h 1 c"), rearrange(f, "w c -> 1 1 w c")
        u = center_pad(field.u, [0, int(N_pad / 2), int(N_pad / 2), 0])
        phase = -jnp.pi * L**2 * (fx**2 + fy**2)

    # Propagating field
    # Propagation phase factor of exp(ij*k*z) is omitted to improve the
    # computation efficiency. This factor does not affect the intensity or
    # the relative phase within the field. Only the absolute phase of the field
    # is affected. - gschlafly
    u = ifft(fft(u, loop_axis) * jnp.exp(1j * phase), loop_axis)

    # Cropping output field
    if mode == "full":
        field = field.replace(u=u)
    elif mode == "same":
        if field.u.ndim > 4:
            u = center_crop(u, [0, 0, int(N_pad / 2), int(N_pad / 2), 0])
        else:
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
        ConcretizationError will arise when traced!).
    """
    # Calculating propagator
    f = []
    for d in range(field.dx.size):
        f.append(jnp.fft.fftfreq(field.shape[1] + N_pad, d=field.dx[..., d].squeeze()))
    f = jnp.stack(f, axis=-1)
    fx, fy = rearrange(f, "h c -> 1 h 1 c"), rearrange(f, "w c -> 1 1 w c")
    kernel = 1 - (field.spectrum / n) ** 2 * (fx**2 + fy**2)
    kernel = jnp.maximum(kernel, 0.0)  # removing evanescent waves
    phase = 2 * jnp.pi * (z * n / field.spectrum) * jnp.sqrt(kernel)

    # Propagating field
    u = center_pad(field.u, [0, N_pad // 2, N_pad // 2, 0])
    u = ifft(fft(u, loop_axis) * jnp.exp(1j * phase), loop_axis)

    # Cropping output field
    if mode == "full":
        field = field.replace(u=u)
    elif mode == "same":
        u = center_crop(u, [0, N_pad // 2, N_pad // 2, 0])
        field = field.replace(u=u)
    else:
        raise NotImplementedError('Only "full" and "same" are supported.')

    return field


def propagate(
    field: Field,
    z: float,
    n: float,
    *,
    method: str = "transfer",
    mode: str = "full",
    N_pad: Optional[int] = None,
    loop_axis: Optional[int] = None,
) -> Field:
    """
    Propagate ``field`` by a distance ``z`` with appropriate padding.

    Allows for propagation with one of three different methods:
        - ``"transform"``: Uses Fresnel transform propagation
        - ``"transfer"``: Uses Fresnel transfer propagation
        - ``"exact"``: Uses exact transfer propagation with no Fresnel approximation

    Args:
        field: ``Field`` to be propagated.
        z: A float that defines the distance to propagate.
        n: A float that defines the refractive index of the medium.
        method: A string that defines the method of propagation.
        mode: A string that defines whether the result is cropped or not. Can
            be either ``"full"`` or ``"same"``.
        N_pad: A keyword argument integer defining the pad length for
            the propagation FFT (NOTE: should not be a Jax array, otherwise a
            ConcretizationError will arise when traced!). If not provided,
            will be calculated automatically based on the spacing and shape
            of the ``field``, the distance to propagate, and the chosen method
            of propagation.
    """
    # Only works for square fields?
    D = field.u.shape[1] * field.dx  # height of field in real coordinates
    Nf = jnp.max((D / 2) ** 2 / (field.spectrum * z))  # Fresnel number
    M = field.u.shape[1]  # height of field in pixels
    # TODO(dd): we should figure out a better approximation method, perhaps by
    # running a quick simulation and checking the aliasing level
    Q = 2 * jnp.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2

    if method == "transform":
        if N_pad is None:
            N = int(jnp.ceil((Q * M) / 2) * 2)
            N_pad = int((N - M))
        field = transform_propagate(field, z, n, N_pad=N_pad, loop_axis=loop_axis)
    elif method == "transfer":
        if N_pad is None:
            N = int(jnp.ceil((Q * M) / 2) * 2)
            N_pad = int((N - M))
        field = transfer_propagate(
            field, z, n, N_pad=N_pad, loop_axis=loop_axis, mode=mode
        )
    elif method == "exact":
        if N_pad is None:
            scale = jnp.max((field.spectrum / (2 * field.dx)))
            assert scale < 1, "Can't do exact transfer when dx < lambda / 2"
            Q = Q / jnp.sqrt(1 - scale**2)  # minimum pad ratio for exact transfer
            N = int(jnp.ceil((Q * M) / 2) * 2)
            N_pad = int((N - M))
        field = exact_propagate(
            field, z, n, N_pad=N_pad, loop_axis=loop_axis, mode=mode
        )
    else:
        raise NotImplementedError(
            "Method must be one of 'transform', 'transfer', or 'exact'."
        )
    return field
