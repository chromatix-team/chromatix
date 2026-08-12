from warnings import warn

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

from chromatix.utils import l2_sq_norm
from chromatix.utils.shapes import _broadcast_1d_to_innermost_batch
from chromatix.utils.initializers import (
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
    "high_na_tube_lens",
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
    z: ScalarLike | Float[Array, "z"] | None = 0.0,
) -> ScalarField | VectorField:
    warn(
        "high_na_ff_lens is deprecated; use high_na_tube_lens",
        DeprecationWarning,
    )
    return high_na_tube_lens(
        field, f, n, NA, z=z, output_shape=output_shape, output_dx=output_dx
    )


def high_na_tube_lens(
    field: ScalarField | VectorField,
    f: float,
    n: float,
    NA: float,
    output_shape: tuple[int, int] | None = None,
    output_dx: ScalarLike | None = None,
    z: ScalarLike | Float[Array, "z"] | None = 0.0,
) -> ScalarField | VectorField:
    """
    Applies a tube lens with apodization of the pupil from the corresponding
    objective lens placed a distance ``f`` (objective focal length) after
    the incoming ``Field``, and allows both defocusing of the input away
    from the focal plane by ``z`` as well as resampling to a desired
    output shape and sampling. This is useful for computing high NA PSFs,
    and is meant to be used with a flat plane wave or Gaussian beam as
    the input (representing the back focal plane of an objective due to
    a point source at the focal plane of the objective). If you want to
    defocus using ``z`` to compute a 3D PSF, the input should be a 2D
    field (no pre-existing z axis). Note that you must ensure that you
    have the appropriate sampling at the input field. See the [high NA PSF
    example](https://chromatix.readthedocs.io/en/latest/examples/highNA_PSF/).

    This model is due to https://arxiv.org/abs/2502.03170.

    This function allows for zooming of the output result by controlling the
    output shape and spacing (dx) via the Chirp Z-transform.

    !!!warning
        This function assumes that the incoming ``Field`` contains only a single
        wavelength and has a square shape.

    !!!warning
        This function assumes that the incoming ``Field`` is from the back
        focal plane of an objective lens. If you are trying to calculate a PSF
        using this function (e.g. with a plane wave as the input), you must
        make sure that the input field has the correct sampling according to
        https://arxiv.org/abs/2502.03170. The input field MUST have a diameter
        greater than or equal to the diameter of the pupil of the objective,
        i.e. ``2 * f * NA / n``. The input field should be appropriately sampled
        at this diameter by choosing a sufficiently high number of pixels.

    !!!warning
        This function assumes that if you are computing multiple defocus planes
        (``z``) that you have a 2D incoming field (no pre-existing z/batch
        axis).

    Args:
        field: The monochromatic ``Field`` to which the lens will be applied.
        f: Focal length of the corresponding objective lens (NOT this tube
            lens; you can choose the magnification by setting\ ``output_dx`` and
            ``output_shape``).
        n: The refractive index of the surrounding medium, e.g. oil.
        NA: The NA of the corresponding objective lens (NOT this tube lens).
        output_shape: The shape of the camera (in pixels). If not provided, the
            output shape will be the same as the shape of the incoming field.
        output_dx: The pixel pitch of the camera (in units of distance). If not
            provided, the output spacing will be the same as the spacing of the
            incoming field.
        z: Defocus distance(s) from the focal plane at which to evaluate the
            field, in units of distance. May be a scalar or a 1D array to
            produce a z-stack. Defaults to ``0.0`` (the focal plane).

    Returns:
        The ``Field`` propagated a distance ``f`` after the tube lens (at the image plane).
    """
    if not isinstance(field, Vector):
        spherical_u = field.u
    else:
        spherical_u = cartesian_to_spherical(field, n, NA, f)
    if output_dx is None:
        output_dx = field.central_dx
    if output_shape is None:
        output_shape = field.spatial_shape
    z = jnp.atleast_1d(jnp.asarray(z)).squeeze()
    if z.size > 1:
        z = _broadcast_1d_to_innermost_batch(z, field.spatial_dims)
    # TODO: This only works for single wavelength so far?
    # TODO: What about non-square cases?
    dk = (
        2
        * jnp.pi
        * n
        * field.central_dx
        * output_dx
        / (field.broadcasted_wavelength * f)
    )
    k_start = -dk * (output_shape[0] // 2)
    k_end = k_start + dk * (output_shape[0] - 1)
    cos_theta = jnp.sqrt(jnp.maximum(1 - l2_sq_norm(field.grid) / f**2, 0.0))
    k = -2 * jnp.pi * n / field.broadcasted_wavelength
    # TODO(dd/2026-08-11): Maybe there should be an optical_fft analogue of
    # zoomed_fft to handle the same kinds of normalizations.
    norm = -1j * n * jnp.prod(field.dx, axis=-1) / (field.broadcasted_wavelength * f)
    correction = jnp.where(
        cos_theta != 0.0,
        norm * jnp.exp(1j * k * cos_theta * z) / cos_theta,
        0.0,
    )
    u = zoomed_fft(
        x=spherical_u * correction,
        k_start=k_start,
        k_end=k_end,
        output_shape=output_shape,
        include_end=True,
        axes=field.spatial_dims,
    )
    output_dx = output_dx * jnp.ones_like(field.dx)
    output_field = field.replace(u=u, dx=output_dx)
    # NOTE(dd/2026-08-11): The double-scaling below is intentional, and we
    # don't use field.extent here so that the extent used to calculate the
    # output_phase below is consistent for both even and odd sizes. We also
    # have to change the centering of the grid so that zero falls on the N // 2
    # center for the same even/odd issue.
    input_extent = field.dx * (2 * (jnp.asarray(field.spatial_shape) // 2))
    output_grid = output_field.grid + 0.5 * (
        jnp.asarray(output_field.spatial_shape) % 2
    ) * output_field.dx
    # NOTE(dd/2026-07-30): The pupil is sampled on a grid centered in the
    # middle at N // 2, but the CZT applies a phase ramp from index 0 in each
    # dimension. This output phase correction factor removes that ramp so we get
    # a clean, centered phase profile (e.g. when simulating a PSF with a plane
    # wave as the input).
    output_phase = (
        jnp.pi
        * n
        / (output_field.broadcasted_wavelength * f)
        * jnp.sum(input_extent * output_grid, axis=-1)
    )
    return output_field.replace(u=output_field.u * jnp.exp(1j * output_phase))


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
