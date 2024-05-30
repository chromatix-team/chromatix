import jax.numpy as jnp
from chromatix.utils import _squeeze_grid_to_2d
from chromatix.utils.fft import fft, ifft
from chromatix import Field
from typing import Union


def optical_fft(field: Field, z: Union[float, Array], n: float) -> Field:
    """
    Computes the optical ``fft`` or ``ifft`` on an incoming ``Field`` propagated
    by ``z``, depending on the sign of ``z`` (which may be an array containing
    both positive and negative values). If ``z`` is positive an ``fft```will
    be performed, otherwise an ``ifft`` (due to the ``1 / (lambda * z)`` term
    in the single Fourier transform Fresnel propagation, which requires this
    behavior). The ``ifft`` is calculated in terms of the conjugate of the
    ``fft`` with appropriate normalization applied so that propagating forwards
    and then backwards yields the same ``Field`` up to numerical precision. This
    function also appropriately changes the sampling of the ``Field`` that is
    output (after propagating to some distance(s) ``z``).
    Args:
        field: The ``Field`` to be propagated by ``fft``.
        z: The distance the ``Field`` will be propagated, assumed to be
            broadcasted to the innermost batch dimension.
        n: Refractive index of the propagation medium.
    Returns:
        The propagated ``Field``, transformed by ``fft``/``ifft``.
    """
    L_sq = field.spectrum * z / n
    du = field.dk * jnp.abs(L_sq)
    # Shift by one pixel in x and y when z < 0 and we ifft
    shifter = jnp.exp(
        1j * (1 * field.k_grid[0] + 1 * field.k_grid[1])
    )
    # Forward transform normalization for z >= 0
    norm_fft = (
        (z >= 0) * -1j * jnp.prod(field.dx, axis=0, keepdims=False) / L_sq
    )
    # Inverse transform normalization for z < 0
    norm_ifft = (
        (z < 0)
        * -1j  # Sign change because we take the conjugate of the input
        * (
            L_sq / jnp.prod(du, axis=0, keepdims=False)  # Inverse length scale
        )
        / jnp.prod(
            jnp.array(field.shape)  # Due to a different norm factor for fft and ifft
        )
    )
    # Inverse transform input needs to use the conjugate and pre-apply a one pixel shift
    fft_input = (norm_fft * field.u) + (norm_ifft * field.conj.u * shifter)
    fft_output = fft(fft_input, axes=field.spatial_dims, shift=True)
    u = (z >= 0) * fft_output + (z < 0) * jnp.conj(fft_output)
    return field.replace(u=u, _dx=_squeeze_grid_to_2d(du, field.ndim))
