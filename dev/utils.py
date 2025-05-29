import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float


def grid(shape: tuple[int, int], spacing: Array) -> Float[Array, "y x d"]:
    N_y, N_x = shape
    dx = rearrange(spacing, "... d -> ... 1 1 d")
    grid = jnp.meshgrid(
        jnp.linspace(0, (N_y - 1), N_y) - N_y / 2,
        jnp.linspace(0, (N_x - 1), N_x) - N_x / 2,
        indexing="ij",
    )
    return dx * jnp.stack(grid, axis=-1)


def freq_grid(shape: tuple[int, int], spacing: Array) -> Float[Array, "y x d"]:
    N_y, N_x = shape
    dk = rearrange(1 / spacing, "... d -> ... 1 1 d")
    grid = jnp.meshgrid(
        jnp.fft.fftshift(jnp.fft.fftfreq(N_y)),
        jnp.fft.fftshift(jnp.fft.fftfreq(N_x)),
        indexing="ij",
    )
    return dk * jnp.stack(grid, axis=-1)


def promote_dx(dx):
    dx = jnp.asarray(dx)
    match dx.size:
        case 1:
            dx = jnp.stack([dx, dx])
        case 2:
            dx = dx
        case _:
            raise ValueError(f"dx must be of size 1 or 2, got {dx.size}.")
    dx = eqx.error_if(dx, jnp.any(dx < 0), f"dx must be larger than 0, got {dx}.")
    return dx
