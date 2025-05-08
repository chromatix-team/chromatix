import jax.numpy as jnp
from jaxtyping import Array

from chromatix import Field


def apply_shift(x: Array, shift: Array) -> Array:
    """Shift is in pixels."""
    n = x.shape[-1]
    x = jnp.pad(x, n // 2, "symmetric")
    f = jnp.fft.fftfreq(2 * n)
    f = jnp.stack(jnp.meshgrid(f, f, indexing="ij"), axis=-1)
    s = jnp.exp(-2 * jnp.pi * 1j * jnp.sum(f * shift, axis=-1))
    return jnp.fft.ifft2(s * jnp.fft.fft2(x))[n // 2 : n + n // 2, n // 2 : n + n // 2]


def shift_field(field: Field, shift: Array) -> Field:
    # TODO: move to field?
    return field.replace(u=apply_shift(field.u.squeeze(), shift)[None, ..., None, None])


