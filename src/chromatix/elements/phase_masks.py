import jax.numpy as jnp
from flax import linen as nn
from chex import Array, PRNGKey
from jax.scipy.ndimage import map_coordinates
from typing import Callable, Optional, Tuple, Union
from ..field import Field
from ..functional import wrap_phase, phase_change
from ..utils import seidel_aberrations, zernike_aberrations
from chromatix.elements.utils import register

__all__ = [
    "PhaseMask",
    "SpatialLightModulator",
    "SeidelAberrations",
    "ZernikeAberrations",
]


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

    Since phase mask initializations might require information about the
    pupil of a system, the extra parameters ``n``, ``f``, and ``NA`` can
    be specified. These will be passed as arguments to the phase mask
    initialization function if ``phase`` is trainable. Note that if any of
    these is None, none of them will be passed to the initialization function
    and you will get an error.

    Attributes:
        phase: The phase to be applied. Should have shape `(H W)`.
        f: Focal length of the system's objective. Defaults to None.
        n: Refractive index of the system's objective. Defaults to None.
        NA: The numerical aperture of the system's objective. Defaults to None.
    """

    phase: Union[Array, Callable[[PRNGKey, Tuple[int, int], float, float], Array]]
    f: Optional[float] = None
    n: Optional[float] = None
    NA: Optional[float] = None

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies ``phase`` mask to incoming ``Field``."""
        if all(x is not None for x in [self.n, self.f, self.NA]):
            pupil_args = (self.n, self.f, self.NA)
        else:
            pupil_args = ()

        phase = register(
            self,
            "phase",
            field.spatial_shape,
            field.dx[..., 0, 0].squeeze(),
            field.spectrum[..., 0, 0].squeeze(),
            *pupil_args,
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
        phase: The phase to be applied. Should have shape `(H W)`.
        shape: The shape of the SLM, provided as (H W).
        spacing: The pitch of the SLM pixels.
        phase_range: The phase range that the SLM can simulate, provided as
            (min, max).
        interpolation_order: The order of interpolation for the SLM pixels to
            the shape of the incoming ``Field``. Can be 0 or 1. Defaults to 0.
        f: Focal length of the system's objective. Defaults to None.
        n: Refractive index of the system's objective. Defaults to None.
        NA: The numerical aperture of the system's objective. Defaults to None.
    """

    phase: Union[Array, Callable[[PRNGKey, Tuple[int, int], float, float], Array]]
    shape: Tuple[int, int]
    spacing: float
    phase_range: Tuple[float, float]
    interpolation_order: int = 0
    f: Optional[float] = None
    n: Optional[float] = None
    NA: Optional[float] = None

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies simulated SLM ``phase`` mask to incoming ``Field``."""
        if all(x is not None for x in [self.n, self.f, self.NA]):
            pupil_args = (self.n, self.f, self.NA)
        else:
            pupil_args = ()

        phase = register(
            self,
            "phase",
            self.shape,
            self.spacing,
            field.spectrum[..., 0, 0].squeeze(),
            *pupil_args,
        )
        assert (
            phase.shape == self.shape
        ), "Provided phase shape should match provided SLM shape"
        phase = wrap_phase(phase, self.phase_range)
        field_pixel_grid = jnp.meshgrid(
            jnp.linspace(0, self.shape[0] - 1, num=field.spatial_shape[0]) + 0.5,
            jnp.linspace(0, self.shape[1] - 1, num=field.spatial_shape[1]) + 0.5,
            indexing="ij",
        )
        phase = map_coordinates(phase, field_pixel_grid, self.interpolation_order)

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

    The ``coefficients`` can be learned by using ``chromatix.utils.trainable``.

    Attributes:
        coefficients: The Seidel coefficients. Should have shape `[5,]`.
        f: The focal length.
        n: The refractive index.
        NA: The numerical aperture. The applied phase will be 0 outside NA.
        u: The horizontal position of the object field point
        v: The vertical position of the object field point
    """

    coefficients: Union[Array, Callable[[PRNGKey], Array]]
    f: float
    n: float
    NA: float
    u: float
    v: float

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies ``phase`` mask to incoming ``Field``."""
        coefficients = register(self, "coefficients")
        phase = seidel_aberrations(
            field.spatial_shape,
            field.dx[..., 0, 0].squeeze(),
            field.spectrum[..., 0, 0].squeeze(),
            self.n,
            self.f,
            self.NA,
            coefficients,
            self.u,
            self.v,
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
        coefficients: The Zernike coefficients as a 1D array.
        f: The focal length.
        n: The refractive index.
        NA: The numerical aperture. The applied phase will be 0 outside NA.
        ansi_indices: Indices of Zernike polynomials (ANSI indexing). Should
            have same length as coefficients.
    """

    coefficients: Union[Array, Callable[[PRNGKey], Array]]
    f: float
    n: float
    NA: float
    ansi_indices: Array

    @nn.compact
    def __call__(self, field: Field) -> Field:
        """Applies ``phase`` mask to incoming ``Field``."""
        coefficients = register(self, "coefficients")

        phase = zernike_aberrations(
            field.spatial_shape,
            field.dx[..., 0, 0].squeeze(),
            field.spectrum[..., 0, 0].squeeze(),
            self.n,
            self.f,
            self.NA,
            self.ansi_indices,
            coefficients,
        )

        return phase_change(field, phase)
