from chromatix import Field
import jax.numpy as jnp
from typing import Optional, Tuple
from chex import Array


def optical_fft(
    field: Field,
    z: float,
    n: float,
    loop_axis: Optional[int] = None,
) -> Field:
    """
    Computes the optical ``fft`` on an incoming ``Field`` propagated by ``z``.

    This means that this function appropriately changes the sampling of the
    ``Field`` that is output (after propagating to some distance ``z``), and
    also computes the correct ``fftshift``/``ifftshift`` as needed.

    Optionally, this function can also compute the ``ifft`` instead (which is
    useful to prevent outputs from being flipped if that is not desired).

    Args:
        field: The ``Field`` to be propagated by ``fft``.
        z: The distance the ``Field`` will be propagated.
        n: Refractive index.

    Returns:
        The propagated ``Field``, transformed by ``fft``/``ifft``.
    """
    # Preliminaries
    L = jnp.sqrt(jnp.complex64(field.spectrum * z / n))  # Lengthscale L
    norm = jnp.abs(field.dx / L) ** 2  # normalization factor

    # finding new coordinates
    du = jnp.abs(L) ** 2 / (field.shape[field.spatial_dims[1]] * field.dx)

    u = -1j * norm * fftshift(fft(ifftshift(field.u, axes=field.spatial_dims), axes=field.spatial_dims, loop_axis=loop_axis), axes=field.spatial_dims)
    return field.replace(u=u, dx=du)


def fftshift(x: Array, axes: Tuple[int, int] = (1, 2)) -> Array:
    """Computes appropriate ``fftshift`` for ``x`` of shape `(B H W C)`."""
    return jnp.fft.fftshift(x, axes=axes)


def ifftshift(x: Array, axes: Tuple[int, int] = (1, 2)) -> Array:
    """Computes appropriate ``ifftshift`` for ``x`` of shape `(B H W C)`."""
    return jnp.fft.ifftshift(x, axes=axes)


def fft(x: Array, axes: Tuple[int, int] = (1, 2), loop_axis: Optional[int] = None) -> Array:
    """Computes ``fft2`` for input of shape `(B H W C)`."""
    if loop_axis is None:
        return jnp.fft.fft2(x, axes=axes)
    else:
        return looped_fft(x, axes, loop_axis)


def ifft(x: Array, axes: Tuple[int, int] = (1, 2), loop_axis: Optional[int] = None) -> Array:
    """Computes ``ifft2`` for input of shape `(B H W C)`."""
    if loop_axis is None:
        return jnp.fft.ifft2(x, axes=axes)
    else:
        return looped_ifft(x, axes, loop_axis)


def looped_fft(x: Array, axes: Tuple[int, int] = (1, 2), loop_axis: int = -1) -> Array:
    # TODO(dd): Fix for any axes or remove altogether
    source = (loop_axis, *axes)
    dest = (0, -2, -1)
    x_fft = jnp.stack([jnp.fft.fft2(y) for y in jnp.moveaxis(x, source, dest)])
    return jnp.moveaxis(x_fft, dest, source)


def looped_ifft(x: Array, axes: Tuple[int, int] = (1, 2), loop_axis: int = -1) -> Array:
    # TODO(dd): Fix for any axes or remove altogether
    source = (loop_axis, *axes)
    dest = (0, -2, -1)
    x_fft = jnp.stack([jnp.fft.ifft2(y) for y in jnp.moveaxis(x, source, dest)])
    return jnp.moveaxis(x_fft, dest, source)
