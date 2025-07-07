import jax.numpy as jnp

from chromatix import Field
from chromatix.field import ScalarField, VectorField, cartesian_to_spherical
from chromatix.functional.amplitude_masks import amplitude_change
from chromatix.functional.convenience import optical_fft
from chromatix.functional.phase_masks import phase_change
from chromatix.functional.rays import (
    compute_free_space_abcd,
    compute_plano_convex_spherical_lens_abcd,
    ray_transfer,
)
from chromatix.typing import Array, ScalarLike
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
    Applies a thin lens placed directly after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` directly after the lens.
    """
    L = jnp.sqrt(field.spectrum * f / n)
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
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

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
    field: Field,
    f: float,
    n: float,
    NA: float,
    output_shape: tuple[int, int] | None = None,
    output_dx: float | None = None,
) -> Field:
    """
    Applies a high NA lens placed a distance ``f`` after the incoming ``Field``.

    !!!warning
        This function assumes that the incoming ``Field`` contains only a single
        wavelength and has a square shape.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        output_shape: The shape of the camera (in pixels). If not provided, the
            output shape will be the same as the shape of the incoming field.
        output_dx: The pixel pitch of the camera (in units of distance). If not
            provided, the output spacing will be the same as the spacing of the
            incoming field.

    Returns:
        The ``Field`` propagated a distance ``f`` to and after the lens.
    """

    if field.shape[-1] == 1:
        # Scalar
        spherical_u = field.u
        create = ScalarField.create
    else:
        # Vectorial
        spherical_u = cartesian_to_spherical(field, n, NA, f)
        create = VectorField.create

    if output_dx is None:
        output_dx = field.dx.squeeze()

    if output_shape is None:
        output_shape = field.spatial_shape

    # NOTE: This only works for single wavelength so far?
    # NOTE: What about non-square cases?
    fov_out = output_shape[0] * output_dx
    zoom_factor = 2 * NA * fov_out / ((field.shape[1] - 1) * field.spectrum.squeeze())

    # Correction factors
    s_grid = field.k_grid * field.spectrum.squeeze() / n
    sz_sq = 1 - NA**2 * l2_sq_norm(s_grid)
    sz = jnp.sqrt(jnp.maximum(sz_sq, 0.0))
    k = 2 * jnp.pi * n / field.spectrum.squeeze()
    defocus = jnp.where(sz != 0.0, jnp.exp(1j * k * sz * f) / sz, 0.0)

    u = zoomed_fft(
        x=spherical_u * defocus,
        k_start=-zoom_factor * jnp.pi,
        k_end=zoom_factor * jnp.pi,
        output_shape=output_shape,
        include_end=True,
        axes=field.spatial_dims,
    )

    return create(output_dx, field.spectrum, field.spectral_density, u)


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
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

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
    L = jnp.sqrt(jnp.complex64(field.spectrum * f / n))  # Lengthscale L
    phase = jnp.pi * (1 - d / f) * l2_sq_norm(field.grid) / jnp.abs(L) ** 2
    return field * jnp.exp(1j * phase)


def microlens_array(
    field: Field,
    n: ScalarLike,
    fs: Array,
    centers: Array,
    radii: Array,
    block_between: bool = False,
) -> Field:
    amplitude, phase = microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
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
    n: ScalarLike,
    f: Array,
    num_lenses_per_side: ScalarLike,
    radius: Array,
    separation: ScalarLike,
    block_between: bool = False,
) -> Field:
    amplitude, phase = hexagonal_microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
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
    num_lenses_height: ScalarLike,
    num_lenses_width: ScalarLike,
    radius: Array,
    separation: ScalarLike,
    block_between: bool = False,
) -> Field:
    amplitude, phase = rectangular_microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
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
    R: ScalarLike,
    center_thickness: ScalarLike,
    n_lens: ScalarLike,
    n_medium: ScalarLike = 1.0,
    NA: ScalarLike | None = None,
    inverse: bool = False,
    magnification: ScalarLike = 1.0,
) -> Field:
    if NA is not None:
        D = 2 * f * NA / n_medium  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    ABCD = compute_plano_convex_spherical_lens_abcd(
        f, R, center_thickness, n_lens, n_medium, inverse
    )
    field = ray_transfer(field, ABCD, n_medium, magnification=magnification)
    return field


def thick_plano_convex_ff_lens(
    field: Field,
    f: ScalarLike,
    R: ScalarLike,
    center_thickness: ScalarLike,
    n_lens: ScalarLike,
    n_medium: ScalarLike = 1.0,
    NA: ScalarLike | None = None,
    inverse: bool = False,
    magnification: ScalarLike = 1.0,
) -> Field:
    if NA is not None:
        D = 2 * f * NA / n_medium  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    _lens = compute_plano_convex_spherical_lens_abcd(
        f, R, center_thickness, n_lens, n_medium, inverse
    )
    _free_space = compute_free_space_abcd(f)
    ABCD = _free_space @ _lens @ _free_space
    field = ray_transfer(field, ABCD, n_medium, magnification=magnification)
    return field
