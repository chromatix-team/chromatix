from typing import Sequence

import equinox as eqx
from chex import assert_rank
from jaxtyping import Array, Float, ScalarLike

from chromatix import Field
from chromatix.functional import interpolated_phase_change, phase_change
from chromatix.typing import m
from chromatix.utils import seidel_aberrations, zernike_aberrations

__all__ = [
    "SeidelAberrations",
    "ZernikeAberrations",
    "PhaseMask",
    "SpatialLightModulator",
]


class PhaseMask(eqx.Module):
    """
    Perturbs ``field`` by ``phase`` (**in radians**), i.e. ``field * exp(1j * phase)``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    Also scales the phase by the ratio of each wavelength to the central
    wavelength of the spectrum (i.e. the first wavelength in the array
    of wavelengths) by default. This scaling is necessary to achieve the
    proper chromatic dispersion effect after propagating a field that is not
    monochromatic through a phase mask (especially if the phase mask is a fine
    grating). Returns a new ``Field`` with the result of the perturbation.

    The ``phase`` can be optimized (pixel by pixel).

    Common phase mask generators can be found in
    [``chromatix.utils.initializers``](utils.md#chromatix.utils.initializers).
    These generators typically assume that the phase mask is placed at the pupil
    plane of a system.

    Attributes:
        phase: The phase mask to apply as a 2D array of phase values (in
            radians) of shape ``(height width)``.
    """

    phase: Float[Array, "h w"]

    def __init__(self, phase: Float[Array, "h w"]):
        self.phase = phase

    def __call__(self, field: Field) -> Field:
        """Applies ``phase`` mask to incoming ``Field``."""
        return phase_change(field, self.phase)


class SpatialLightModulator(eqx.Module):
    """
    Simulates a spatial light modulator (SLM) applied to an incoming ``Field``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    This means that this element acts as if the SLM is phase only and transmits
    a ``Field``, rather than reflecting it.

    Also scales the phase by the ratio of each wavelength to the central
    wavelength of the spectrum (i.e. the first wavelength in the array
    of wavelengths) by default. This scaling is necessary to achieve the
    proper chromatic dispersion effect after propagating a field that is not
    monochromatic through a phase mask (especially if the phase mask is a fine
    grating). Returns a new ``Field`` with the result of the perturbation.

    This element also handles the limited phase modulation range of an SLM,
    controlled by ``phase_range``.

    This element also approximates the simulation of the surface of the SLM by
    interpolating the phase with the given ``shape`` (which can be the number of
    pixels in the SLM) to the shape of the incoming ``Field``, which is assumed
    to have a shape larger than the given SLM ``shape``. The order of the
    interpolation performed can be controlled with ``interpolation_order``, but
    currently only orders of 0 (nearest neighbor) and 1 (linear interpolation)
    are supported.

    Attributes:
        phase: The phase mask to apply as a 2D array of phase values (in
            radians) of shape ``(height width)``.
        shape: The shape of the SLM in number of pixels, provided as `(height
            width)`.
        spacing: The pitch of the SLM pixels in units of distance.
        phase_range: The phase range in radians that the SLM can simulate,
            provided as `(min, max)`.
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
    """

    phase: Float[Array, "h w"]
    shape: tuple[int, int]
    spacing: ScalarLike
    phase_range: Float[Array, "2"] | tuple[float, float]
    num_bits: int | None = None
    interpolation_order: int = 0

    def __init__(
        self,
        phase: Float[Array, "h w"],
        shape: tuple[int, int],
        spacing: ScalarLike,
        phase_range: Float[Array, "2"] | tuple[float, float],
        num_bits: int | None = None,
        interpolation_order: int = 0,
    ):
        self.phase = phase
        assert_rank(
            self.phase, 2, custom_message="Phase must be a 2D array of shape (H W)"
        )
        assert self.phase.shape == shape, (
            f"Phase array must have same shape as SLM: expected {shape} got {self.phase.shape}"
        )
        self.shape = shape
        self.spacing = spacing
        self.phase_range = phase_range
        self.num_bits = num_bits
        self.interpolation_order

    def __call__(self, field: Field) -> Field:
        """Applies simulated SLM ``phase`` mask to incoming ``Field``."""
        return interpolated_phase_change(
            field, self.phase, self.phase_range, self.num_bits, self.interpolation_order
        )


class SeidelAberrations(eqx.Module):
    """
    Applies Seidel phase polynomial to an incoming ``Field``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    This element handles multi-wavelength ``Field``s by assuming that the first
    wavelength in the ``spectrum`` of the ``Field`` is the central wavelength
    for which the ``phase`` was calculated, and modulates the ``phase`` by the
    ratio of other wavelengths in the ``spectrum`` to the central wavelength
    appropriately.

    Attributes:
        coefficients: The Seidel coefficients. Should have shape `(5,)`.
        f: The focal length of the system's objective lens in units of distance.
        n: The refractive index of the medium.
        NA: The numerical aperture. The applied phase will be 0 outside NA.
    """

    coefficients: Float[Array, "5"]
    f: ScalarLike
    n: ScalarLike
    NA: ScalarLike

    def __init__(
        self,
        coefficients: Float[Array, "5"],
        f: ScalarLike,
        n: ScalarLike,
        NA: ScalarLike,
    ):
        self.coefficients = coefficients
        self.f = f
        self.n = n
        self.NA = NA

    def __call__(self, field: Field, u: ScalarLike, v: ScalarLike) -> Field:
        """
        Applies ``phase`` mask to incoming ``Field``.

        Args:
            field: The complex field to be perturbed.
            u: The horizontal position of the object field point in normalized
                coordinates from 0 to +/- 1. A value of 0 represents the center
                coordinate in the plane while a value of 1 represents the farthest
                point from the center. Positive values go right and negative values
                go left.
            v: The vertical position of the object field point in normalized
                coordinates from 0 to +/- 1. A value of 0 represents the center
                coordinate in the plane while a value of 1 represents the farthest
                point from the center. Positive values go down and negative values
                go up.
        """
        phase = seidel_aberrations(
            field.spatial_shape,
            field.central_dx,
            field.central_wavelength,
            self.n,
            self.f,
            self.NA,
            self.coefficients,
            self.u,
            self.v,
        )
        return phase_change(field, phase)


class ZernikeAberrations(eqx.Module):
    """
    Applies Zernike aberrations to an incoming ``Field``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    This element handles multi-wavelength ``Field``s by assuming that the first
    wavelength in the ``spectrum`` of the ``Field`` is the central wavelength
    for which the ``phase`` was calculated, and modulates the ``phase`` by the
    ratio of other wavelengths in the ``spectrum`` to the central wavelength
    appropriately.

    Attributes:
        coefficients: The Zernike coefficients as a 1D array of the same length as the
            ``ansi_indices``.
        f: The focal length of the system's objective lens in units of distance.
        n: The refractive index of the medium.
        NA: The numerical aperture. The applied phase will be 0 outside NA.
        ansi_indices: Linear Zernike indices according to ANSI numbering.
        coefficients: Weight coefficients for the Zernike polynomials.
        normalize: Whether to normalize the Zernike coefficients. Defaults to
            ``True``.
    """

    coefficients: Float[Array, "m"]
    f: ScalarLike
    n: ScalarLike
    NA: ScalarLike
    ansi_indices: Sequence[int]
    normalize: bool

    def __init__(
        self,
        coefficients: Float[Array, "m"],
        f: ScalarLike,
        n: ScalarLike,
        NA: ScalarLike,
        ansi_indices: Sequence[int],
        normalize: bool = True,
    ):
        self.coefficients = coefficients
        self.f = f
        self.n = n
        self.NA = NA
        self.ansi_indices = ansi_indices
        self.normalize = normalize

    def __call__(self, field: Field) -> Field:
        """Applies ``phase`` mask to incoming ``Field``."""
        phase = zernike_aberrations(
            field.spatial_shape,
            field.central_dx,
            field.central_wavelength,
            self.n,
            self.f,
            self.NA,
            self.ansi_indices,
            self.coefficients,
            self.normalize,
        )
        return phase_change(field, phase)
