import jax.numpy as jnp
from chromatix import Field
from functools import partial
from chex import Array


def optical_fft(field: Field, z: float, n: float) -> Field:
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

    u = -1j * norm * fft(field.u, shift=True)
    du = field.dk * jnp.abs(L) ** 2  # new spacing
    return field.replace(u=u, dx=du)


def fft(x: Array, shift: bool = False) -> Array:
    """Computes ``fft2`` for input of shape `[B H W C]`.
    If shift is true, first applies ifftshift, than an fftshift to
    make sure everything stays centered."""
    fft = partial(jnp.fft.fft2, axes=[1, 2])
    fftshift = partial(jnp.fft.fftshift, axes=[1, 2])
    ifftshift = partial(jnp.fft.ifftshift, axes=[1, 2])

    if shift:
        return fftshift(fft(ifftshift(x)))
    else:
        return fft(x)


def ifft(x: Array, shift: bool = False) -> Array:
    """Computes ``ifft2`` for input of shape `[B H W C]`.
    If shift is true, first applies ifftshift, than an fftshift to
    make sure everything stays centered."""
    ifft = partial(jnp.fft.ifft2, axes=[1, 2])
    fftshift = partial(jnp.fft.fftshift, axes=[1, 2])
    ifftshift = partial(jnp.fft.ifftshift, axes=[1, 2])

    if shift:
        return fftshift(ifft(ifftshift(x)))
    else:
        return ifft(x)
