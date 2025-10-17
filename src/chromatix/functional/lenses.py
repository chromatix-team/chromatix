import jax.numpy as jnp
from jaxtyping import Array, Float, ScalarLike

from chromatix import Field, ScalarField, Vector, VectorField, cartesian_to_spherical
from chromatix.functional.amplitude_masks import amplitude_change
from chromatix.functional.convenience import optical_fft
from chromatix.functional.phase_masks import phase_change
from chromatix.functional.rays import (
    compute_free_space_abcd,
    compute_plano_convex_spherical_lens_abcd,
    ray_transfer,
)
from chromatix.typing import m
from chromatix.utils.czt import zoomed_fft

from ..utils import l2_sq_norm
from ..utils.initializers import (
    hexagonal_microlens_array_amplitude_and_phase,
    microlens_array_amplitude_and_phase,
    rectangular_microlens_array_amplitude_and_phase,
)
from .pupils import circular_pupil

__all__ = [
    "thin_lens",
    "ff_lens",
    "df_lens",
    "microlens_array",
    "hexagonal_microlens_array",
    "rectangular_microlens_array",
    "thick_plano_convex_lens",
    "thick_plano_convex_ff_lens",
    "high_na_ff_lens",
]


def thin_lens(
    field: Field, f: ScalarLike, n: ScalarLike, NA: ScalarLike | None = None
) -> Field:
    """
    Applies a thin lens placed immediately in the plane of the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens in units of distance.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` immediately after the lens.
    """
    L = jnp.sqrt(field.broadcasted_wavelength * f / n)
    phase = -jnp.pi * l2_sq_norm(field.grid) / L**2

    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    return field * jnp.exp(1j * phase)


def ff_lens(
    field: Field,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike | None = None,
    inverse: bool = False,
) -> Field:
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens in units of distance.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether the field is passing forwards or backwards through
            the lens. If ``True``, the phase of the lens is conjugated.
            Defaults to ``False``.

    Returns:
        The ``Field`` propagated a distance ``f`` to and after the lens.
    """
    # Pupil
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    if inverse:
        # if inverse, propagate over negative distance
        f = -f
    return optical_fft(field, f, n)


def high_na_ff_lens(
    field: ScalarField | VectorField,
    f: float,
    n: float,
    NA: float,
    output_shape: tuple[int, int] | None = None,
    output_dx: ScalarLike | None = None,
) -> ScalarField | VectorField:
    """
    Applies a high NA lens placed a distance ``f`` after the incoming ``Field``.

    !!!warning
        This function assumes that the incoming ``Field`` contains only a single
        wavelength and has a square shape.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        NA: The NA of the lens.
        output_shape: The shape of the camera (in pixels). If not provided, the
            output shape will be the same as the shape of the incoming field.
        output_dx: The pixel pitch of the camera (in units of distance). If not
            provided, the output spacing will be the same as the spacing of the
            incoming field.

    Returns:
        The ``Field`` propagated a distance ``f`` to and after the lens.
    """
    if not isinstance(field, Vector):
        spherical_u = field.u
    else:
        spherical_u = cartesian_to_spherical(field, n, NA, f)
    if output_dx is None:
        output_dx = field.central_dx
    if output_shape is None:
        output_shape = field.spatial_shape
    # TODO: This only works for single wavelength so far?
    # TODO: What about non-square cases?
    fov_out = output_shape[0] * output_dx
    zoom_factor = 2 * NA * fov_out / ((field.shape[1] - 1) * field.spectrum.wavelength)
    # Correction factors
    s_grid = field.f_grid * field.spectrum.wavelength / n
    sz_sq = 1 - NA**2 * l2_sq_norm(s_grid)
    sz = jnp.sqrt(jnp.maximum(sz_sq, 0.0))
    k = 2 * jnp.pi * n / field.spectrum.wavelength
    defocus = jnp.where(sz != 0.0, jnp.exp(1j * k * sz * f) / sz, 0.0)
    # Create zoomed field
    u = zoomed_fft(
        x=spherical_u * defocus,
        k_start=-zoom_factor * jnp.pi,
        k_end=zoom_factor * jnp.pi,
        output_shape=output_shape,
        include_end=True,
        axes=field.spatial_dims,
    )
    output_dx = output_dx * jnp.ones_like(field.dx)
    return field.replace(u=u, dx=output_dx)


def df_lens(
    field: Field,
    d: ScalarLike,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike | None = None,
    inverse: bool = False,
) -> Field:
    """
    Applies a thin lens placed a distance ``d`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        d: Distance from the incoming ``Field`` to the lens.
        f: Focal length of the lens in units of distance.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether the field is passing forwards or backwards through
            the lens. If ``True``, the phase of the lens is conjugated.
            Defaults to ``False``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    if inverse:
        # if inverse, propagate over negative distance
        f = -d
        d = -f
    field = optical_fft(field, f, n)

    # Phase factor due to distance d from lens
    L = jnp.sqrt(jnp.complex64(field.broadcasted_wavelength * f / n))  # Lengthscale L
    phase = jnp.pi * (1 - d / f) * l2_sq_norm(field.grid) / jnp.abs(L) ** 2
    return field * jnp.exp(1j * phase)


def microlens_array(
    field: Field,
    fs: Float[Array, "m"],
    n: ScalarLike,
    centers: Float[Array, "m"],
    radii: Float[Array, "m"],
    block_between: bool = False,
) -> Field:
    """
    Applies a microlens array of arbitrary positioned microlenses placed
    immediately in the plane of the incoming ``Field``.

    !!!warning
        If you have recently used this function prior to it being documented,
        note that the arguments have changed.

    Args:
        field: The ``Field`` to which the lens will be applied.
        fs: A 1D array of shape ``(lenses)`` defining the focal lengths of each
            lens in units of distance.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        centers: A 2D array of shape ``(lenses 2)`` defining the center position
            in units of distance (in `y x` order) for each lens of the microlens
            array.
        radii: A 1D array of shape ``(lenses)`` defining the radius of each
            microlens in units of distance.

    Returns:
        The ``Field`` immediately after the microlens array.
    """
    amplitude, phase = microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field.central_dx,
        field.central_wavelength,
        n,
        fs,
        centers,
        radii,
    )
    field = phase_change(field, phase)
    if block_between:
        field = amplitude_change(field, amplitude)
    return field


def hexagonal_microlens_array(
    field: Field,
    f: ScalarLike,
    n: ScalarLike,
    num_lenses_per_side: int,
    radius: Array,
    separation: ScalarLike,
    block_between: bool = False,
) -> Field:
    """
    Applies a microlens array of hexagonally arranged microlenses placed
    immediately in the plane of the incoming ``Field``.

    !!!warning
        If you have recently used this function prior to it being documented,
        note that the arguments have changed.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: A scalar value defining the focal length of each lens in units of
            distance.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        num_lenses_per_side: The number of lenses on each outer side of the
            hexagon (e.g. setting this number to 4 will create 37 microlenses).
        radius: A scalar value defining the radius of each microlens in units
            of distance.
        separation: A scalar value defining how far apart the center of each
            microlens is from its neighbors in units of distance.
        block_between: If ``True``, will mask out the ``Field`` in the spaces
            between the microlenses. For example, this is useful to suppress
            background from light that is not focused by the microlenses in the
            PSF of a Fourier light-field microscope. Defaults to ``False``, in
            which case no blocking of light occurs.

    Returns:
        The ``Field`` immediately after the microlens array.
    """
    amplitude, phase = hexagonal_microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field.central_dx,
        field.central_wavelength,
        n,
        f,
        num_lenses_per_side,
        radius,
        separation,
    )
    field = phase_change(field, phase)
    if block_between:
        field = amplitude_change(field, amplitude)
    return field


def rectangular_microlens_array(
    field: Field,
    n: ScalarLike,
    f: Array,
    num_lenses_height: int,
    num_lenses_width: int,
    radius: Array,
    separation: ScalarLike,
    block_between: bool = False,
) -> Field:
    """
    Applies a microlens array of hexagonally arranged microlenses placed
    immediately in the plane of the incoming ``Field``.

    !!!warning
        If you have recently used this function prior to it being documented,
        note that the arguments have changed.

    Args:
        field: The ``Field`` to which the lens will be applied.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        f: A scalar value defining the focal length of each lens in units of
            distance.
        num_lenses_height: The number of lenses on each vertical side of the
            rectangle.
        num_lenses_width: The number of lenses on each horizontal side of the
            rectangle.
        radius: A scalar value defining the radius of each microlens in units
            of distance.
        separation: A scalar value defining how far apart the center of each
            microlens is from its neighbors in units of distance.
        block_between: If ``True``, will mask out the ``Field`` in the spaces
            between the microlenses. For example, this is useful to suppress
            background from light that is not focused by the microlenses in the
            PSF of a Fourier light-field microscope. Defaults to ``False``, in
            which case no blocking of light occurs.

    Returns:
        The ``Field`` immediately after the microlens array.
    """
    amplitude, phase = rectangular_microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field.central_dx,
        field.central_wavelength,
        n,
        f,
        num_lenses_height,
        num_lenses_width,
        radius,
        separation,
    )
    field = phase_change(field, phase)
    if block_between:
        field = amplitude_change(field, amplitude)
    return field


def thick_plano_convex_lens(
    field: Field,
    f: ScalarLike,
    radius: ScalarLike,
    center_thickness: ScalarLike,
    n_lens: ScalarLike,
    n_medium: ScalarLike = 1.0,
    NA: ScalarLike | None = None,
    inverse: bool = False,
    magnification: ScalarLike = 1.0,
) -> Field:
    """
    Applies a thick plano-convex lens placed immediately in the plane of the
    incoming ``Field``. This lens includes propagation by a small distance
    within the lens (defined by ``center_thickness``).

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: The focal length of the lens in units of distance.
        radius: The radius of the spherical part of the plano-convex lens in
            units of distance.
        center_thickness: The maximum thickness of the plano-convex lens (i.e.
            the distance through the center of the lens) in units of distance.
        n_lens: The refractive index of the lens material (e.g. glass).
        n_medium: The refractive index of the surrounding medium (assumed to be
            the same incoming and exiting). Defaults to 1.0 for air.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether the field is passing forwards (plano-convex) or
            backwards (convex-plano) through the lens. If ``True``, the phase of
            the lens is conjugated. Defaults to ``False``.
        magnification: The magnification to be applied to the propagation
            through the system. A magnification of greater than 1 will zoom
            in during the propagation (decrease the spacing of the outgoing
            ``Field``) and a magnification of smaller than 1 will do the
            opposite. Defaults to 1.0 for no change to the spacing of the
            ``Field``.

    Returns:
        The ``Field`` immediately after the lens.
    """
    if NA is not None:
        D = 2 * f * NA / n_medium  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    ABCD = compute_plano_convex_spherical_lens_abcd(
        f, radius, center_thickness, n_lens, n_medium, inverse
    )
    field = ray_transfer(field, ABCD, n_medium, magnification=magnification)
    return field


def thick_plano_convex_ff_lens(
    field: Field,
    f: ScalarLike,
    radius: ScalarLike,
    center_thickness: ScalarLike,
    n_lens: ScalarLike,
    n_medium: ScalarLike = 1.0,
    NA: ScalarLike | None = None,
    inverse: bool = False,
    magnification: ScalarLike = 1.0,
) -> Field:
    """
    Applies a thick plano-convex lens placed a distance ``f`` after the incoming
    ``Field``. This lens includes propagation by a small distance within the
    lens (defined by ``center_thickness``).

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: The focal length of the lens in units of distance.
        radius: The radius of the spherical part of the plano-convex lens in
            units of distance.
        center_thickness: The maximum thickness of the plano-convex lens (i.e.
            the distance through the center of the lens) in units of distance.
        n_lens: The refractive index of the lens material (e.g. glass).
        n_medium: The refractive index of the surrounding medium (assumed to be
            the same incoming and exiting). Defaults to 1.0 for air.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether the field is passing forwards (plano-convex) or
            backwards (convex-plano) through the lens. If ``True``, the phase of
            the lens is conjugated. Defaults to ``False``.
        magnification: The magnification to be applied to the propagation
            through the system. A magnification of greater than 1 will zoom
            in during the propagation (decrease the spacing of the outgoing
            ``Field``) and a magnification of smaller than 1 will do the
            opposite. Defaults to 1.0 for no change to the spacing of the
            ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    if NA is not None:
        D = 2 * f * NA / n_medium  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    _lens = compute_plano_convex_spherical_lens_abcd(
        f, radius, center_thickness, n_lens, n_medium, inverse
    )
    _free_space = compute_free_space_abcd(f)
    ABCD = _free_space @ _lens @ _free_space
    field = ray_transfer(field, ABCD, n_medium, magnification=magnification)
    return field
