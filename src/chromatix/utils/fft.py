from functools import partial

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike


def fft(x: ArrayLike, axes: tuple[int, int] = (1, 2), shift: bool = False) -> Array:
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


def ifft(x: ArrayLike, axes: tuple[int, int] = (1, 2), shift: bool = False) -> Array:
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
