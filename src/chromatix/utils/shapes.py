import equinox as eqx
import jax.numpy as jnp
from einops import rearrange, repeat
from jaxtyping import Array, ArrayLike, Float, ScalarLike

from chromatix.typing import wv

__all__ = [
    "_broadcast_1d_to_channels",
    "_broadcast_1d_to_polarization",
    "_broadcast_1d_to_innermost_batch",
    "_broadcast_1d_to_grid",
    "_broadcast_2d_to_grid",
    "_broadcast_dx_to_grid",
    "_squeeze_grid_to_2d",
    "_broadcast_2d_to_spatial",
]


def _verify_spatial_dims(spatial_dims: tuple[int, int]):
    assert len(spatial_dims) == 2, "Spatial dims must be tuple of length 2"
    assert spatial_dims[1] == spatial_dims[0] + 1, "Spatial dims must be contiguous"
    assert all(s < 0 for s in spatial_dims), "Spatial dims must be negative indices"


def _broadcast_1d_to_channels(
    wv: Float[Array, "wv"], spatial_dims: tuple[int, int]
) -> Array:
    """Broadcast 1D array of size `wavelengths` to `(... height width wavelengths *vectorial)`."""
    _verify_spatial_dims(spatial_dims)
    if wv.size > 1:
        assert spatial_dims[-1] != -1, (
            "Cannot broadcast multiple wavelengths to monochromatic Field"
        )
        shape_spec = "wv -> "
        for i in range(abs(spatial_dims[0])):
            if i != 2:
                shape_spec += "1 "
            else:
                shape_spec += "wv "
        return rearrange(jnp.atleast_1d(wv), shape_spec)
    else:
        return wv.squeeze()


def _broadcast_1d_to_polarization(x: ScalarLike, ndim: int) -> Array:
    """Broadcast 1D array of size `P` to `(B... H W C [1 | 3])`."""
    shape_spec = "p -> " + ("1 " * (ndim - 1)) + "p"
    return rearrange(jnp.atleast_1d(x), shape_spec)


def _broadcast_1d_to_innermost_batch(
    x: ScalarLike, spatial_dims: tuple[int, int]
) -> Array:
    """Broadcast 1D array of size `B` to left of `(H W)` in `(B... H W C [1 | 3])`."""
    _verify_spatial_dims(spatial_dims)
    shape_spec = "b -> b " + "1 " * abs(spatial_dims[0])
    return rearrange(jnp.atleast_1d(x), shape_spec)


def _broadcast_1d_to_grid(x: ArrayLike | tuple[float, float], ndim: int) -> Array:
    """
    Broadcast 1D array of size `2` to `(2 B... H W C 1)`.
    Useful for vectorial ops on grids.
    """
    shape_spec = "d ->" + "d" + " 1" * (ndim - 4) + " 1 1 1 1"
    return rearrange(jnp.atleast_1d(jnp.array(x)), shape_spec, d=2)


def _broadcast_2d_to_grid(x: Array, ndim: int) -> Array:
    """
    Broadcast 2D array of shape `2 C` to `(2 B... H W C [1 | 3])`.
    Useful for vectorial ops on grids.
    """
    shape_spec = "d c ->" + "d" + " 1" * (ndim - 4) + " 1 1 c 1"
    return rearrange(x, shape_spec, d=2)


def _broadcast_dx_to_grid(dx: float | Array, num_wavelengths: int) -> Array:
    """
    Broadcast 2D array of shape `(channels 2)` to `(wavelengths 2)`.
    Useful for vectorial ops on grids.
    """
    dx = jnp.atleast_1d(jnp.asarray(dx))
    match dx.shape, num_wavelengths:
        case (1,), wv:
            dx = repeat(dx, "1 -> wv d", wv=wv, d=2)
        case (_,), wv:
            assert dx.shape[0] == wv, "Number of wavelengths does not match"
            dx = repeat(dx, "wv -> wv d", wv=wv, d=2)
        case (_, 2), wv:
            assert dx.shape[0] == wv, "Number of wavelengths does not match"
            dx = dx
        case _:
            raise ValueError(
                f"dx must be scalar or have shape (2,), (C,), or (C, 2); got {dx.size}."
            )
    dx = eqx.error_if(dx, jnp.any(dx < 0), f"dx must be larger than 0, got {dx}.")
    return dx


def _squeeze_grid_to_2d(x: Array, ndim: int) -> Array:
    """
    Squeeze array of shape `(2 B... H W C [1 | 3])` to 2D array of shape `2 C`.
    Useful for vectorial ops on grids.
    """
    shape_spec = "d" + " 1" * (ndim - 4) + " 1 1 c 1 -> d c"
    return rearrange(x, shape_spec, d=2)


def _broadcast_2d_to_spatial(x: Array, spatial_dims: tuple[int, int]) -> Array:
    """Broadcast 2D array of shape `(H W)` to `(H W [*C] [*1])`."""
    _verify_spatial_dims(spatial_dims)
    shape_spec = "h w -> h w " + ("1 " * (abs(spatial_dims[0]) - 2))
    return rearrange(x, shape_spec)
