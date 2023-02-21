from chromatix import Field
import jax.numpy as jnp
from typing import Optional
from chex import Array


def fftshift(x: Array) -> Array:
    return jnp.fft.fftshift(x, axes=[1, 2])


def ifftshift(x: Array) -> Array:
    return jnp.fft.ifftshift(x, axes=[1, 2])


def optical_fft(
    field: Field,
    z: float,
    n: float,
    loop_axis: Optional[int] = None,
    inverse: bool = False,
) -> Field:
    # Preliminaries
    L = jnp.sqrt(field.spectrum * z / n)  # Lengthscale L
    norm = (field.dx / L) ** 2  # normalization factor

    # finding new coordinates
    du = L ** 2 / (field.shape[1] * field.dx)

    # Doing the FFT
    if inverse:
        u = norm * fftshift(fft(field.u, loop_axis))
    else:
        u = norm * ifftshift(ifft(field.u, loop_axis))
    return -1j * field.replace(u=u, dx=du)


def fft(x: Array, loop_axis=None) -> Array:
    if loop_axis is None:
        return jnp.fft.fft2(x, axes=[1, 2])
    else:
        return looped_fft(x, loop_axis)


def ifft(x: Array, loop_axis=None) -> Array:
    if loop_axis is None:
        return jnp.fft.ifft2(x, axes=[1, 2])
    else:
        return looped_ifft(x, loop_axis)


def looped_fft(x: Array, loop_axis: int) -> Array:
    # Given array of shape [n, x, y, m], loops over first or last axis
    # and does fft2 on [x, y].
    source = (loop_axis, 1, 2)
    dest = (0, -2, -1)
    x_fft = jnp.stack([jnp.fft.fft2(y) for y in jnp.moveaxis(x, source, dest)])
    return jnp.moveaxis(x_fft, dest, source)


def looped_ifft(x: Array, loop_axis: int) -> Array:
    # Given array of shape [n, x, y, m], loops over first or last axis
    # and does fft2 on [x, y].
    source = (loop_axis, 1, 2)
    dest = (0, -2, -1)
    x_fft = jnp.stack([jnp.fft.ifft2(y) for y in jnp.moveaxis(x, source, dest)])
    return jnp.moveaxis(x_fft, dest, source)
