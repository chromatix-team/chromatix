import jax.numpy as jnp
from chromatix.utils import _squeeze_grid_to_2d
from chromatix.utils.fft import fft
from chromatix import Field


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
    L = jnp.sqrt(jnp.complex64(field.spectrum * z / n))  # Lengthscale L
    norm = jnp.prod(field.dx, axis=0, keepdims=False) / jnp.abs(L) ** 2
    u = -1j * norm * fft(field.u, axes=field.spatial_dims, shift=True)
    du = field.dk * jnp.abs(L) ** 2  # New spacing
    return field.replace(u=u, _dx=_squeeze_grid_to_2d(du, field.ndim))
