import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, ScalarLike

from chromatix import Field
from chromatix.utils import l2_sq_norm
from chromatix.utils.fft import fft, ifft

__all__ = [
    "ray_transfer",
    "compute_free_space_abcd",
    "compute_thin_spherical_lens_abcd",
    "compute_plano_convex_spherical_lens_abcd",
]


def compute_free_space_abcd(
    z: ScalarLike,
) -> Array:
    """
    Computes the ABCD (ray transfer) matrix for free-space propagation.

    Args:
        z: The distance to propagate (in units of distance) through free space
            (or through a homogenous medium of constant scalar refractive
            index).

    Returns:
        The ``Field`` directly after the propagation.
    """
    ABCD = jnp.array([[1, z], [0, 1]])
    return ABCD


def compute_thin_spherical_lens_abcd(
    f: ScalarLike,
    inverse: bool = False,
) -> Array:
    """
    Computes the ABCD (ray transfer) matrix for a thin spherical lens applied
    immediately at the current plane of the field.

    Args:
        f: The focal length of the lens in units of distance.
        inverse: Whether the field is passing forwards or backwards through
            the lens. If ``True``, the phase of the lens is negated. Defaults
            to ``False``.

    Returns:
        The ``Field`` directly after the lens.
    """
    if inverse:
        f = -f
    ABCD = jnp.array([[1, 0], [-1 / f, 1]])
    return ABCD


def compute_plano_convex_spherical_lens_abcd(
    f: ScalarLike,
    radius: ScalarLike,
    center_thickness: ScalarLike,
    n_lens: ScalarLike,
    n_medium: ScalarLike = 1.0,
    inverse: bool = False,
) -> Array:
    """
    Computes the ABCD (ray transfer) matrix for a thick plano-convex spherical
    lens applied immediately at the current plane of the field. This matrix is
    calculated such that the rays exit after propagating a small distance within
    the lens (defined by ``center_thickness``).

    Args:
        f: The focal length of the lens in units of distance.
        radius: The radius of the spherical part of the plano-convex lens in
            units of distance.
        center_thickness: The maximum thickness of the plano-convex lens (i.e.
            the distance through the center of the lens) in units of distance.
        n_lens: The refractive index of the lens material (e.g. glass).
        n_medium: The refractive index of the surrounding medium (assumed to be
            the same incoming and exiting). Defaults to 1.0 for air.
        inverse: Whether the field is passing forwards (plano-convex) or
            backwards (convex-plano) through the lens. If ``True``, the phase of
            the lens is negated. Defaults to ``False``.

    Returns:
        The ``Field`` directly after the lens.
    """
    _center = jnp.array([[1, center_thickness], [0, 1]])
    if inverse:
        _entrance = jnp.array([[1, 0], [0, n_medium / n_lens]])
        _exit = jnp.array(
            [[1, 0], [(n_lens - n_medium) / (-radius * n_medium), n_lens / n_medium]]
        )
    else:
        _entrance = jnp.array(
            [[1, 0], [(n_medium - n_lens) / (radius * n_lens), n_medium / n_lens]]
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
    """
    Applies the ``ABCD`` (ray transfer) matrix to the incoming ``Field`` and
    returns the outgoing ``Field``. Uses Collins' integral [1] to propagate the
    ``Field`` through the system described by the transfer matrix.

    [1]: S. Collins, "Lens-System Diffraction Integral Written in Terms of
    Matrix Optics*," J. Opt. Soc. Am. 60, 1168-1177 (1970).

    Args:
        field: The incoming ``Field`` to which the transfer matrix should be applied.
        ABCD: The ray transfer matrix defining how a paraxial incoming ray is
            perturbed through the system described by the transfer matrix.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        magnification: The magnification to be applied to the propagation
            through the system. A magnification of greater than 1 will zoom
            in during the propagation (decrease the spacing of the outgoing
            ``Field``) and a magnification of smaller than 1 will do the
            opposite. Defaults to 1.0 for no change to the spacing of the
            ``Field``.

    Returns:
        The ``Field`` directly after the system described by the ray transfer matrix.
    """
    A = ABCD[0, 0]
    B = ABCD[0, 1]
    D = ABCD[1, 1]
    k = 2 * jnp.pi * n / field.broadcasted_wavelength
    input_phase = k / (2 * B) * (A - magnification) * l2_sq_norm(field.grid)
    transfer_phase = (
        jnp.pi * (field.broadcasted_wavelength / n) * B / magnification
    ) * l2_sq_norm(field.f_grid)
    output_phase = (
        k
        / (2 * B)
        * (D - 1 / magnification)
        * l2_sq_norm(field.grid / (magnification**2))
    )
    fft_input = field.u * jnp.exp(1j * input_phase)
    axes = field.spatial_dims
    u = jnp.exp(1j * output_phase) * ifft(
        fft(fft_input, axes=axes, shift=True) * jnp.exp(-1j * transfer_phase),
        axes=axes,
        shift=True,
    )
    field = field.replace(u=u, dx=field.dx / magnification)
    return field
