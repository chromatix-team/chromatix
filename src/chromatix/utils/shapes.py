from einops import rearrange
from chex import Array
from typing import Union
import jax.numpy as jnp

__all__ = [
    "_broadcast_1d_to_channels",
    "_broadcast_1d_to_innermost_batch",
    "_broadcast_2d_to_spatial",
]


def _broadcast_1d_to_channels(x: Union[float, Array], ndim: int) -> Array:
    """Broadcast 1D array of size `C` to `(B... H W C)`."""
    shape_spec = "c -> " + ("1 " * (ndim - 1)) + "c"
    return rearrange(jnp.atleast_1d(x), shape_spec)


def _broadcast_1d_to_innermost_batch(x: Union[float, Array], ndim: int) -> Array:
    """Broadcast 1D array of size `B` to left of `(H W)` in `(B... H W C)`."""
    shape_spec = "z ->" + " 1" * (ndim - 4) + " z 1 1 1"
    return rearrange(jnp.atleast_1d(x), shape_spec)


def _broadcast_2d_to_spatial(x: Array, ndim: int) -> Array:
    """Broadcast 2D array of shape `(H W)` to `(B... H W C)`."""
    shape_spec = "h w ->" + ("1 " * (ndim - 3)) + "h w 1"
    return rearrange(x, shape_spec)
