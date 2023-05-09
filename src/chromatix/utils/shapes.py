from einops import rearrange
from chex import Array
from typing import Union
import jax.numpy as jnp

__all__ = [
    "_broadcast_1d_to_channels",
    "_broadcast_1d_to_polarization",
    "_broadcast_1d_to_innermost_batch",
    "_broadcast_1d_to_grid",
    "_broadcast_2d_to_grid",
    "_squeeze_grid_to_2d",
    "_broadcast_2d_to_spatial",
]


def _broadcast_1d_to_channels(x: Union[float, Array], ndim: int) -> Array:
    """Broadcast 1D array of size `C` to `(B... H W C [1 | 3])`."""
    shape_spec = "c -> " + ("1 " * (ndim - 2)) + "c 1"
    return rearrange(jnp.atleast_1d(x), shape_spec)


def _broadcast_1d_to_polarization(x: Union[float, Array], ndim: int) -> Array:
    """Broadcast 1D array of size `P` to `(B... H W C [1 | 3])`."""
    shape_spec = "p -> " + ("1 " * (ndim - 1)) + "p"
    return rearrange(jnp.atleast_1d(x), shape_spec)


def _broadcast_1d_to_innermost_batch(x: Union[float, Array], ndim: int) -> Array:
    """Broadcast 1D array of size `B` to left of `(H W)` in `(B... H W C [1 | 3])`."""
    shape_spec = "b ->" + " 1" * (ndim - 5) + " b 1 1 1 1"
    return rearrange(jnp.atleast_1d(x), shape_spec)


def _broadcast_1d_to_grid(x: Union[float, Array], ndim: int) -> Array:
    """
    Broadcast 1D array of size `2` to `(2 B... H W C 1)`.
    Useful for vectorial ops on grids.
    """
    shape_spec = "d ->" + "d" + " 1" * (ndim - 4) + " 1 1 1 1"
    return rearrange(jnp.atleast_1d(x), shape_spec, d=2)


def _broadcast_2d_to_grid(x: Union[float, Array], ndim: int) -> Array:
    """
    Broadcast 2D array of shape `2 C` to `(2 B... H W C [1 | 3])`.
    Useful for vectorial ops on grids.
    """
    shape_spec = "d c ->" + "d" + " 1" * (ndim - 4) + " 1 1 c 1"
    return rearrange(x, shape_spec, d=2)


def _squeeze_grid_to_2d(x: Union[float, Array], ndim: int) -> Array:
    """
    Squeeze array of shape `(2 B... H W C [1 | 3])` to 2D array of shape `2 C`.
    Useful for vectorial ops on grids.
    """
    shape_spec = "d" + " 1" * (ndim - 4) + " 1 1 c 1 -> d c"
    return rearrange(x, shape_spec, d=2)


def _broadcast_2d_to_spatial(x: Array, ndim: int) -> Array:
    """Broadcast 2D array of shape `(H W)` to `(B... H W C [1 | 3])`."""
    if x.ndim != ndim:
        shape_spec = "h w ->" + ("1 " * (ndim - 4)) + "h w 1 1"
        return rearrange(x, shape_spec)
    else:
        return x
