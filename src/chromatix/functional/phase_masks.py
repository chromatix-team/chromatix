import jax
import jax.numpy as jnp
from chex import assert_equal_shape
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, ArrayLike, Float, ScalarLike

from chromatix import Field, Spectrum
from chromatix.ops.quantization import quantize
from chromatix.utils.initializers import (
    axicon_phase,
    linear_phase,
    sawtooth_phase,
    sinusoid_phase,
)
from chromatix.utils.shapes import _broadcast_2d_to_spatial

__all__ = [
    "phase_change",
    "interpolated_phase_change",
    "wrap_phase",
    "spectrally_modulate_phase",
    "thin_prism",
    "sawtooth_grating",
    "sinusoid_grating",
    "axicon",
]


def phase_change(
    field: Field, phase: Float[Array, "h w"], spectrally_modulate: bool = True
) -> Field:
    """
    Perturbs ``field`` by ``phase`` (**in radians**), i.e. ``field * exp(1j * phase)``.

    Also scales the phase by the ratio of each wavelength to the central
    wavelength of the spectrum (i.e. the first wavelength in the array
    of wavelengths) by default. This scaling is necessary to achieve the
    proper chromatic dispersion effect after propagating a field that is not
    monochromatic through a phase mask (especially if the phase mask is a fine
    grating). Returns a new ``Field`` with the result of the perturbation.

    Common phase mask generators can be found in ``chromatix.utils.initializers``.
    These generators typically assume that the phase mask is placed at the pupil
    plane of a system.

    Args:
        field: The complex field to be perturbed.
        phase: The phase mask to apply as a 2D array of phase values (in
            radians) of shape ``(height width)``.
        spectrally_modulate: Whether to perform the per-wavelength scaling
            of the phase based on the given spectrum of the incoming field.
            Assumes the phase mask is designed for the central wavelength of the
            spectrum (assumed to be the first wavelength in the spectrum). Set
            to ``True`` by default.

    Returns:
        The perturbed ``Field`` immediately after the phase mask.
    """
    if phase.ndim != field.ndim:
        phase = _broadcast_2d_to_spatial(phase, field.spatial_dims)
    assert_equal_shape(
        (phase, field.u),
        dims=field.spatial_dims,
        custom_message="Phase must have same height and width as incoming ``Field``.",
    )
    if spectrally_modulate:
        phase = spectrally_modulate_phase(phase, field.spectrum)
    return field * jnp.exp(1j * phase)


@jax.custom_jvp
def wrap_phase(
    phase: Float[Array, "h w"],
    limits: ArrayLike | tuple[float, float] = (-jnp.pi, jnp.pi),
) -> Array:
    """
    Wraps values of ``phase`` to the range given by ``limits``.

    Args:
        phase: The phase mask to wrap (in radians) as a 2D array of shape
            ``(height width)`` (though this function will work on an array of
            any shape).
        limits: A tuple defining the minimum and maximum value that ``phase``
            will be wrapped to.
    """
    phase_min, phase_max = limits
    phase = jnp.where(
        phase < phase_min,
        phase + (2 * jnp.pi * (1 + (phase_min - phase) // (2 * jnp.pi))),
        phase,
    )
    phase = jnp.where(
        phase > phase_max,
        phase - (2 * jnp.pi * (1 + (phase - phase_max) // (2 * jnp.pi))),
        phase,
    )
    return phase


@wrap_phase.defjvp
def wrap_phase_jvp(primals: tuple, tangents: tuple) -> tuple:
    return wrap_phase(*primals), tangents[0]


def spectrally_modulate_phase(phase: Array, spectrum: Spectrum) -> Array:
    """
    Spectrally modulates a given ``phase`` for multiple wavelengths.

    Scales the phase by the ratio of each wavelength to the central
    wavelength of the spectrum (i.e. the first wavelength in the array
    of wavelengths) by default. This scaling is necessary to achieve the
    proper chromatic dispersion effect after propagating a field that is not
    monochromatic through a phase mask (especially if the phase mask is a fine
    grating).

    Args:
        phase: The phase mask to apply as a 2D array of phase values (in
            radians) of shape ``(height width)``.
        spectrum: The ``Spectrum`` used to modulate the phase (contains the
            wavelengths necessary to scale the phase values). The first
            wavelength in the spectrum is assumed to be the central wavelength
            for which the phase mask was designed.
    """
    return phase * spectrum.spectral_modulation


def interpolated_phase_change(
    field: Field,
    phase: Float[Array, "h w"],
    phase_range: tuple[float, float] | None = None,
    num_bits: int | None = None,
    interpolation_order: int = 0,
    spectrally_modulate: bool = True,
) -> Field:
    """
    Perturbs ``field`` by ``phase`` (**in radians**), i.e. ``field * exp(1j
    * phase)``, but with an interpolation of the phase mask to the number of
    samples (pixels) in the incoming field. This is useful when the phase mask
    itself has a different number of pixels than the field (typically the phase
    mask has fewer pixels than the finely-sampled field, e.g. when simulating a
    spatial light modulator). Assumes that the phase mask should be interpolated
    to the entire extent of the incoming field, i.e. that the extent of the
    field is exactly that of the phase mask.

    Also scales the phase by the ratio of each wavelength to the central
    wavelength of the spectrum (i.e. the first wavelength in the array
    of wavelengths) by default. This scaling is necessary to achieve the
    proper chromatic dispersion effect after propagating a field that is not
    monochromatic through a phase mask (especially if the phase mask is a fine
    grating). Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        phase: The phase mask to apply as a 2D array of phase values (in
            radians) of shape ``(height width)``.
        phase_range: A tuple of two scalar values defining the minimum and
            maximum phase range values respectively (in radians) to which the
            provided phase values should be wrapped. For example, this could
            be `(0.0, 2 * jnp.pi)`. This is useful when simulating devices with
            some allowed range of phase values or that have a baseline phase
            offset. Defaults to ``None``, in which case no wrapping of the phase
            mask is performed.
        num_bits: An integer value representing the number of bits to which the
            phase mask should be quantized, i.e. only `2**num_bits` values will
            be allowed in the phase mask. This quantization occurs after the
            wrapping of the phase values to the provided ``phase_range``, or if
            no ``phase_range`` is provided then to the full range of values in
            the provided ``phase``.
        interpolation_order: An integer defining the order of the interpolation
            of the phase values to the number of pixels of the field. Set to `0`
            by default for nearest-neighbor interpolation, but can also be set
            to 1 for bilinear interpolation. No higher order interpolation is
            supported for now.
        spectrally_modulate: Whether to perform the per-wavelength scaling
            of the phase based on the given spectrum of the incoming field.
            Assumes the phase mask is designed for the central wavelength of the
            spectrum (assumed to be the first wavelength in the spectrum). Set
            to ``True`` by default.

    Returns:
        The perturbed ``Field`` immediately after the phase mask.
    """
    if phase_range is not None:
        phase = wrap_phase(phase, phase_range)
    if num_bits is not None:
        phase = quantize(phase, num_bits, range=phase_range)
    # NOTE(dd/2025-08-14): Assumes the phase should be interpolated across the full
    # extent of the Field
    field_pixel_grid = jnp.meshgrid(
        jnp.linspace(0, phase.shape[0] - 1, num=field.spatial_shape[0]) + 0.5,
        jnp.linspace(0, phase.shape[1] - 1, num=field.spatial_shape[1]) + 0.5,
        indexing="ij",
    )
    phase = map_coordinates(phase, field_pixel_grid, interpolation_order)
    return phase_change(field, phase, spectrally_modulate=spectrally_modulate)


def thin_prism(
    field: Field,
    n_prism: ScalarLike,
    max_thickness: ScalarLike,
    rotation: ScalarLike = 0.0,
    n_medium: ScalarLike = 1.0,
) -> Field:
    """
    Applies a thin prism placed immediately in the plane of the incoming ``Field``.

    The prism is applied as a linear phase ramp that starts at 0 and increases
    horizontally to the right to a maximum value determined by the specified
    ``max_thickness``. To change the direction of the prism, a ``rotation`` can
    be applied.

    Args:
        field: The ``Field`` to which the prism will be applied.
        n_prism: The refractive index of the prism material (e.g. glass).
        max_thickness: A scalar defining the maximum thickness of the prism in
            units of distance.
        rotation: How much to rotate the prism in-plane (counter-clockwise in
            radians).
        n_medium: The refractive index of the surrounding medium. Defaults to
            1.0 for air.

    Returns:
        The perturbed ``Field`` immediately after the prism.
    """
    phase = linear_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n_prism,
        max_thickness,
        rotation,
        n_medium,
    )
    field = phase_change(field, phase)
    return field


def sawtooth_grating(
    field: Field,
    n_grating: ScalarLike,
    period: ScalarLike,
    thickness: ScalarLike,
    rotation: ScalarLike = 0.0,
    n_medium: ScalarLike = 1.0,
) -> Field:
    """
    Applies a sawtooth grating placed immediately in the plane of the incoming
    ``Field``.

    The grating can be varied in thickness which changes the overall scale
    of the phase values used to calculate the effect of the grating (i.e. the
    grating phase is calculated as a thin sample). To change the direction of
    the grating, a ``rotation`` can be applied.

    Args:
        field: The ``Field`` to which the grating will be applied.
        n_grating: The refractive index of the grating material (e.g. glass).
        period: The period of the sawtooth wave defining the grating in units
            of distance.
        thickness: A scalar defining the thickness of the grating in units of
            distance.
        rotation: How much to rotate the grating in-plane (counter-clockwise
            in radians).
        n_medium: The refractive index of the surrounding medium. Defaults to
            1.0 for air.

    Returns:
        The perturbed ``Field`` immediately after the grating.
    """
    phase = sawtooth_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n_grating,
        period,
        thickness,
        rotation,
        n_medium,
    )
    field = phase_change(field, phase)
    return field


def sinusoid_grating(
    field: Field,
    n_grating: ScalarLike,
    period: ScalarLike,
    thickness: ScalarLike,
    rotation: ScalarLike = 0.0,
    n_medium: ScalarLike = 1.0,
) -> Field:
    """
    Applies a sinusoid grating placed immediately in the plane of the incoming
    ``Field``.

    The grating can be varied in thickness which changes the overall scale
    of the phase values used to calculate the effect of the grating (i.e. the
    grating phase is calculated as a thin sample). To change the direction of
    the grating, a ``rotation`` can be applied.

    Args:
        field: The ``Field`` to which the grating will be applied.
        n_grating: The refractive index of the grating material (e.g. glass).
        period: The period of the sinusoid wave defining the grating in units
            of distance.
        thickness: A scalar defining the thickness of the grating in units of
            distance.
        rotation: How much to rotate the grating in-plane (counter-clockwise
            in radians).
        n_medium: The refractive index of the surrounding medium. Defaults to
            1.0 for air.

    Returns:
        The perturbed ``Field`` immediately after the grating.
    """
    phase = sinusoid_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n_grating,
        period,
        thickness,
        rotation,
        n_medium,
    )
    field = phase_change(field, phase)
    return field


def axicon(
    field: Field,
    n_axicon: ScalarLike,
    slope_angle: ScalarLike,
    n_medium: ScalarLike = 1.0,
) -> Field:
    """
    Applies an axicon placed immediately in the plane of the incoming ``Field``.

    The steepness of the axicon can be controlled with ``slope_angle``. To change the
    direction of the prism, a ``rotation`` can be applied.

    Args:
        field: The ``Field`` to which the axicon will be applied.
        n_axicon: The refractive index of the axicon material (e.g. glass).
        slope_angle: The angle between the base of the axicon and the base in
            radians.
        n_medium: The refractive index of the surrounding medium. Defaults to
            1.0 for air.

    Returns:
        The perturbed ``Field`` immediately after the axicon.
    """
    phase = axicon_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n_axicon,
        slope_angle,
        n_medium,
    )
    field = phase_change(field, phase)
    return field
