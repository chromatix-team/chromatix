import jax.numpy as jnp
from chromatix.utils import _squeeze_grid_to_2d
from chromatix.utils.fft import fft, ifft
from chromatix import Field


def conditional_fft(field: Field, z: float, n: float):
    """
    Computes the ``fft`` or th ``ifft`` on an incoming ``Field`` propagated by ``z``, depending on the sign of ``z``.
    If ``z`` is positive an ``fft```will be performed, otherwise an ``ifft``. The reason is that Signle step Fresnel propagation
    has a 1/(lambda z) term in the FFT, which therefore should yield this behaviour.
    Note that the required norm factors are also included in this fft.
    This fairly complicated way of calculating an ifft in terms of an fft is needed, since we want to enable a potential mix of positive and
    negative values for ``z``.
    """
    L_sq = field.spectrum * z / n  # Lengthscale L^2
    du = field.dk * jnp.abs(L_sq)  # New spacing. pixel spacing is always positive
    shifter = jnp.exp(
        1j * (1 * field.k_grid[0] + 1 * field.k_grid[1])
    )  # shifts by one pixel in x and y (all sizes are even!)
    norm_fft = (  # fft for positive z
        (z >= 0) * -1j * jnp.prod(field.dx, axis=0, keepdims=False) / L_sq
    )
    norm_ifft = (  # ifft for negative z, needs to use the conjugate and pre-apply a one pixel shift
        -1j  # also account for the conjugate beeing applied below
        * (z < 0)
        * (  # since all the steps should be run in backwards order we have to devide. However z is negative which changes the sign of norm, requiring the phase change
            L_sq / jnp.prod(du, axis=0, keepdims=False)
        )
        / jnp.prod(
            jnp.array(field.shape)
        )  # due to a different norm factor for fft and ifft
    )
    fft_input = norm_fft * field.u + norm_ifft * field.conj.u * shifter
    fft_output = fft(fft_input, axes=field.spatial_dims, shift=True)
    return (z >= 0) * fft_output + (z < 0) * jnp.conj(fft_output)


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
    L_sq = field.spectrum * z / n  # Lengthscale L^2
    du = field.dk * jnp.abs(L_sq)  # New spacing is always positive
    u = conditional_fft(field, z, n)
    # if z < 0:
    #   norm = jnp.prod(du, axis=0, keepdims=False) / L_sq
    #   u = ifft(
    #     field.u * (1j / norm), axes=field.spatial_dims, shift=True
    #   )  # since all the steps should be run in backwards order we have to devide. However z is negative which changes the sign of norm, requiring the phase change
    # else:
    #   norm = jnp.prod(field.dx, axis=0, keepdims=False) / L_sq
    #   u = -1j * norm * fft(field.u, axes=field.spatial_dims, shift=True)
    return field.replace(u=u, _dx=_squeeze_grid_to_2d(du, field.ndim))
