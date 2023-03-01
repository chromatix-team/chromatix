import jax.numpy as jnp

from ..field import Field
from ..functional.phase_masks import wrap_phase, spectrally_modulate_phase, phase_change, seidel_aberrations, zernike_aberrations
from typing import Callable, Union, Tuple
from einops import rearrange
from flax import linen as nn
from chex import Array, PRNGKey, assert_rank
from jax.scipy.ndimage import map_coordinates
import pdb

__all__ = ["PhaseMask",
           "SpatialLightModulator",
           "SeidelAberrations",
           "ZernikeAberrations"]


class PhaseMask(nn.Module):
    """
    Applies a ``phase`` mask to an incoming ``Field``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    This element handles multi-wavelength ``Field``s by assuming that the first
    wavelength in the ``spectrum`` of the ``Field`` is the central wavelength
    for which the ``phase`` was calculated, and modulates the ``phase`` by the
    ratio of other wavelengths in the ``spectrum`` to the central wavelength
    appropriately.

    The ``phase`` can be learned (pixel by pixel) by using
    ``chromatix.utils.trainable``.

    Attributes:
        phase: The phase to be applied. Should have shape `[1 H W 1]`.
    """

    phase: Union[Array, Callable[[PRNGKey, Tuple[int, ...]], Array]]

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies ``phase`` mask to incoming ``Field``."""
        phase = (
            self.param("phase_pixels", self.phase, (1, *field.shape[1:3], 1))
            if callable(self.phase)
            else self.phase
        )
        assert_rank(phase, 4, custom_message="Phase must be array of shape [1 H W 1]")
        phase = spectrally_modulate_phase(
            phase, field.spectrum, field.spectrum[..., 0].squeeze()
        )
        return phase_change(field, phase)


class SpatialLightModulator(nn.Module):
    """
    Simulates a spatial light modulator (SLM) applied to an incoming ``Field``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    This means that this element acts as if the SLM is phase only and transmits
    a ``Field``, rather than reflecting it.

    This element handles multi-wavelength ``Field``s by assuming that the first
    wavelength in the ``spectrum`` of the ``Field`` is the central wavelength
    for which the ``phase`` was calculated, and modulates the ``phase`` by the
    ratio of other wavelengths in the ``spectrum`` to the central wavelength
    appropriately.

    This element also handles the limited phase modulation range of an SLM,
    controlled by ``phase_range``.

    This element also roughly handles the simulation of the surface of
    the SLM by interpolating the phase with the given ``shape`` (which
    can be the number of pixels in the SLM) to the shape of the incoming
    ``Field``, which is assumed to have a shape larger than the given SLM
    ``shape``. The order of the interpolation performed can be controlled with
    ``interpolation_order``, but currently only orders of 0 (nearest neighbor)
    and 1 (linear interpolation) are supported.

    The ``phase`` of the SLM can be learned (pixel by pixel) by using
    ``chromatix.utils.trainable``.

    Attributes:
        phase: The phase to be applied. Should have shape `[1 H W 1]`.
        shape: The shape of the SLM, provided as (H W).
        phase_range: The phase range that the SLM can simulate, provided as
            (min, max).
        interpolation_order: The order of interpolation for the SLM pixels to
            the shape of the incoming ``Field``. Can be 0 or 1. Defaults to 0.
    """

    phase: Union[Array, Callable[[PRNGKey, Tuple[int, ...]], Array]]
    shape: Tuple[int, int]
    phase_range: Tuple[float, float]
    interpolation_order: int = 0

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies simulated SLM ``phase`` mask to incoming ``Field``."""
        phase = (
            self.param("slm_pixels", self.phase, (1, *self.shape, 1))
            if callable(self.phase)
            else self.phase
        )
        assert_rank(phase, 4, custom_message="Phase must be array of shape [1 H W 1]")
        assert (
            phase.shape[1:3] == self.shape
        ), "Provided phase shape should match provided SLM shape"
        phase = wrap_phase(phase, self.phase_range)
        field_pixel_grid = jnp.meshgrid(
            jnp.linspace(0, self.shape[0] - 1, num=field.shape[1]) + 0.5,
            jnp.linspace(0, self.shape[1] - 1, num=field.shape[2]) + 0.5,
            indexing="ij",
        )
        phase = map_coordinates(
            phase.squeeze(), field_pixel_grid, self.interpolation_order
        )
        phase = rearrange(phase, "h w -> 1 h w 1")
        phase = spectrally_modulate_phase(
            phase, field.spectrum, field.spectrum[..., 0].squeeze()
        )
        return phase_change(field, phase)


class SeidelAberrations(nn.Module):
    """
    Applies Seidel phase polynomial to an incoming ``Field``.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    This element handles multi-wavelength ``Field``s by assuming that the first
    wavelength in the ``spectrum`` of the ``Field`` is the central wavelength
    for which the ``phase`` was calculated, and modulates the ``phase`` by the
    ratio of other wavelengths in the ``spectrum`` to the central wavelength
    appropriately.

    The ``phase`` can be learned (pixel by pixel) by using
    ``chromatix.utils.trainable``.

    Attributes:
        phase: The phase to be applied. Should have shape `[1 H W 1]`.
    """

    coefficients: Union[Array, Callable[[PRNGKey], Array]]
    f: float
    n: float
    NA: float
    u: float
    v:float

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies ``phase`` mask to incoming ``Field``."""
        coefficients = (
            self.param("seidel_coefficients", self.coefficients)
            if callable(self.coefficients)
            else self.coefficients
        )
        #assert_rank(phase, 4, custom_message="Phase must be array of shape [1 H W 1]")
        phase = seidel_aberrations(field.shape, field.dx, field.spectrum[..., 0].squeeze(), self.n, self.f, self.NA, coefficients, self.u, self.v)
        phase = spectrally_modulate_phase(
            phase, field.spectrum, field.spectrum[..., 0].squeeze()
        )
        return phase_change(field, phase)


class ZernikeAberrations(nn.Module):
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
        ansi_indices:
        coefficients: length of coefficients
    """
    
    n: float
    f: float
    NA: float
    ansi_indices: Array
    coefficients: Union[Array, Callable[[PRNGKey], Array]]
    
    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies ``phase`` mask to incoming ``Field``."""
        coefficients = (
            self.param("zernike_coefficients", self.coefficients)
            if callable(self.coefficients)
            else self.coefficients
        )
        
        phase = zernike_aberrations(field.shape, field.dx, field.spectrum[..., 0].squeeze(),
                                    self.n, self.f, self.NA, self.ansi_indices, coefficients)
        phase = spectrally_modulate_phase(
            phase, field.spectrum, field.spectrum[..., 0].squeeze()
        )
        return phase_change(field, phase)