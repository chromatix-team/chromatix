import jax.numpy as jnp
from jax import Array

from chromatix import Field
from chromatix.typing import ArrayLike, ScalarLike
from chromatix.utils import l2_sq_norm
from chromatix.utils.fft import fft, ifft


__all__ = ["ray_transfer", "compute_plano_convex_spherical_lens_abcd"]


def compute_free_space_abcd(
    d: ScalarLike,
) -> Array:
    ABCD = jnp.array([[1, d], [0, 1]])
    return ABCD


def compute_plano_convex_spherical_lens_abcd(
    f: ScalarLike,
    R: ScalarLike,
    center_thickness: ScalarLike,
    n_lens: ScalarLike,
    n_medium: ScalarLike = 1.0,
    inverse: bool = False,
) -> Array:
    _center = jnp.array([[1, center_thickness], [0, 1]])
    if inverse:
        _entrance = jnp.array([[1, 0], [0, n_medium / n_lens]])
        _exit = jnp.array(
            [[1, 0], [(n_lens - n_medium) / (-R * n_medium), n_lens / n_medium]]
        )
    else:
        _entrance = jnp.array(
            [[1, 0], [(n_medium - n_lens) / (R * n_lens), n_medium / n_lens]]
        )
        _exit = jnp.array([[1, 0], [0, n_lens / n_medium]])
    ABCD = _exit @ _center @ _entrance
    return ABCD


def ray_transfer(
    field: Field,
    ABCD: ArrayLike,
    n: ScalarLike,
    magnification: ScalarLike = 1.0,
) -> Field:
    A = ABCD[0, 0]
    B = ABCD[0, 1]
    D = ABCD[1, 1]
    k = 2 * jnp.pi * n / field.spectrum
    input_phase = k / (2 * B) * (A - magnification) * l2_sq_norm(field.grid)
    transfer_phase = (jnp.pi * field.spectrum * B / magnification) * l2_sq_norm(
        field.k_grid
    )
    output_phase = (
        k / (2 * B) * (D - 1 / magnification) * l2_sq_norm(field.grid / magnification)
    )
    fft_input = field.u * jnp.exp(1j * input_phase)
    axes = field.spatial_dims
    u = jnp.exp(1j * output_phase) * ifft(
        fft(fft_input, axes=axes, shift=True) * jnp.exp(-1j * transfer_phase),
        axes=axes,
        shift=True,
    )
    field = field.replace(u=u, _dx=field._dx / magnification)
    return field
