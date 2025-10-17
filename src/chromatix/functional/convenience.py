import jax.numpy as jnp
from einops import rearrange
from jaxtyping import ScalarLike

from chromatix import Field
from chromatix.utils.fft import fft


def optical_fft(field: Field, z: ScalarLike, n: ScalarLike) -> Field:
    """
    Computes the optical ``fft`` or ``ifft`` on an incoming ``Field`` propagated
    by ``z``, assuming that the distance is in the far field. Can also be used
    for simulating propagation through a lens from the focal plane to the back
    focal plane if ``z`` is the focal length of the lens. The direction of the
    propagation depends on the sign of ``z`` (which is a scalar value that may
    be positive or negative). If ``z`` is positive an ``fft```will be performed,
    otherwise an ``ifft`` (due to the ``1 / (lambda * z)`` term in the single
    Fourier transform Fresnel propagation, which requires this behavior).
    The ``ifft`` is calculated in terms of the conjugate of the ``fft`` with
    appropriate normalization applied so that propagating forwards and then
    backwards yields the same ``Field`` up to numerical precision. This function
    also appropriately changes the sampling (``dx``) of the resulting ``Field``.

    Args:
        field: The ``Field`` to be propagated by ``fft``.
        z: Scalar representing how far the ``Field`` will be propagated in units
            of distance.
        n: Real-valued scalar representing (isotropic) refractive index of the
            propagation medium.
    Returns:
        The propagated ``Field``, transformed by ``fft``/``ifft``.
    """
    L_sq = field.broadcasted_wavelength * z / n
    if field.spectrum.size > 1:
        shape_spec = "wv -> wv"
        for i in range(field.df.ndim - 1):
            shape_spec += " 1"
        _L_sq = rearrange(L_sq.squeeze(), shape_spec)
        du = field.df * jnp.abs(_L_sq)
    else:
        du = field.df * jnp.abs(L_sq)
    # Forward transform normalization for z >= 0
    norm_fft = (z >= 0) * -1j * jnp.prod(field.dx, axis=-1) / L_sq
    # Inverse transform normalization for z < 0
    norm_ifft = (
        (z < 0)
        * -1j  # Sign change because we take the conjugate of the input
        * (L_sq / jnp.prod(du, axis=-1))  # Inverse length scale
        / jnp.prod(
            jnp.array(
                field.spatial_shape
            )  # Due to a different norm factor for fft and ifft
        )
    )
    # Inverse transform input needs to use the conjugate
    fft_input = (norm_fft * field.u) + (norm_ifft * field.conj.u)
    fft_output = fft(fft_input, axes=field.spatial_dims, shift=True)
    u = (z >= 0) * fft_output + (z < 0) * jnp.conj(fft_output)
    return field.replace(u=u, dx=du)
