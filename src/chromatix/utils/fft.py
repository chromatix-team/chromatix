import jax.numpy as jnp
from typing import Tuple
from chex import Array
from functools import partial


def fft(x: Array, axes: Tuple[int, int] = (1, 2), shift: bool = False) -> Array:
    """
    Computes ``fft2`` for input of shape `(B... H W C)`.
    If shift is true, first applies ``ifftshift``, than an ``fftshift`` to
    make sure everything stays centered.
    """
    fft = partial(jnp.fft.fft2, axes=axes)
    fftshift = partial(jnp.fft.fftshift, axes=axes)
    ifftshift = partial(jnp.fft.ifftshift, axes=axes)
    if shift:
        return fftshift(fft(ifftshift(x)))
    else:
        return fft(x)


def ifft(x: Array, axes: Tuple[int, int] = (1, 2), shift: bool = False) -> Array:
    """
    Computes ``ifft2`` for input of shape `(B... H W C)`.
    If shift is true, first applies ``ifftshift``, than an ``fftshift`` to
    make sure everything stays centered.
    """
    ifft = partial(jnp.fft.ifft2, axes=axes)
    fftshift = partial(jnp.fft.fftshift, axes=axes)
    ifftshift = partial(jnp.fft.ifftshift, axes=axes)
    if shift:
        return fftshift(ifft(ifftshift(x)))
    else:
        return ifft(x)
