import jax.numpy as jnp
import flax.linen as nn
from ..field import Field, ScalarField, VectorField
from ..functional.sources import (
    plane_wave,
    point_source,
    objective_point_source,
    generic_field,
)
from typing import Optional, Callable, Tuple, Union
from chex import PRNGKey, Array

__all__ = ["PointSource", "ObjectivePointSource", "PlaneWave", "GenericField"]


class PointSource(nn.Module):
    """
    Generates field due to point source a distance ``z`` away.

    Can also be given ``pupil``.

    The attributes ``z``, ``n``, ``power``, and ``amplitude`` can be learned by
    using ``chromatix.utils.trainable``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        z: The distance of the point source.
        n: Refractive index.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarisation.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """

    shape: Tuple[int, int]
    dx: Union[float, Array]
    spectrum: Union[float, Array]
    spectral_density: Union[float, Array]
    z: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    power: Union[float, Callable[[PRNGKey], float]] = 1.0
    amplitude: Union[float, Array, Callable[[PRNGKey], Array]] = 1.0
    pupil: Optional[Callable[[Field], Field]] = None
    scalar: bool = True

    def setup(self):
        self._z = self.param("_z", self.z) if isinstance(self.z, Callable) else self.z
        self._n = self.param("_n", self.n) if isinstance(self.n, Callable) else self.n
        self._power = (
            self.param("_power", self.power)
            if isinstance(self.power, Callable)
            else self.power
        )
        self._amplitude = (
            self.param("_amplitude", self.amplitude)
            if isinstance(self.amplitude, Callable)
            else self.amplitude
        )

    def __call__(self) -> Field:
        return point_source(
            self.shape,
            self.dx,
            self.spectrum,
            self.spectral_density,
            self._z,
            self._n,
            self._power,
            self._amplitude,
            self.pupil,
            self.scalar,
        )


class ObjectivePointSource(nn.Module):
    """
    Generates field due to a point source defocused by an amount ``z`` away from
    the focal plane, just after passing through a lens with focal length ``f``
    and numerical aperture ``NA``.

    The attributes ``f``, ``n``, ``NA``, and ``power`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        f: Focal length of the objective lens.
        n: Refractive index.
        NA: The numerical aperture of the objective lens.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarisation.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """

    shape: Tuple[int, int]
    dx: Union[float, Array]
    spectrum: Union[float, Array]
    spectral_density: Union[float, Array]
    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Union[float, Callable[[PRNGKey], float]]
    power: Union[float, Callable[[PRNGKey], float]] = 1.0
    amplitude: Union[float, Array, Callable[[PRNGKey], Array]] = 1.0
    scalar: bool = True

    def setup(self):
        self._f = self.param("_f", self.f) if isinstance(self.f, Callable) else self.f
        self._n = self.param("_n", self.n) if isinstance(self.n, Callable) else self.n
        self._NA = (
            self.param("_NA", self.NA) if isinstance(self.NA, Callable) else self.NA
        )
        self._power = (
            self.param("_power", self.power)
            if isinstance(self.power, Callable)
            else self.power
        )
        self._amplitude = (
            self.param("_amplitude", self.amplitude)
            if isinstance(self.amplitude, Callable)
            else self.amplitude
        )

    def __call__(self, z: float) -> Field:
        return objective_point_source(
            self.shape,
            self.dx,
            self.spectrum,
            self.spectral_density,
            z,
            self._f,
            self._n,
            self._NA,
            self._power,
            self._amplitude,
            self.scalar,
        )


class PlaneWave(nn.Module):
    """
    Generates plane wave of given ``phase`` and ``power``.

    Can also be given ``pupil`` and ``kykx`` vector to control the angle of the
    plane wave.

    The attributes ``kykx``, ``power``, and ``amplitude`` can be learned by
    using ``chromatix.utils.trainable``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarisation.
        kykx: Defines the orientation of the plane wave. Should be an
            array of shape `[2,]` in the format [ky, kx].
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """

    shape: Tuple[int, int]
    dx: Union[float, Array]
    spectrum: Union[float, Array]
    spectral_density: Union[float, Array]
    power: Union[float, Callable[[PRNGKey], float]] = 1.0
    amplitude: Union[float, Array, Callable[[PRNGKey], Array]] = 1.0
    kykx: Array = jnp.zeros((2,))
    pupil: Optional[Callable[[Field], Field]] = None
    scalar: bool = True

    def setup(self):
        self._kykx = (
            self.param("_kykx", self.kykx)
            if isinstance(self.kykx, Callable)
            else self.kykx
        )
        self._power = (
            self.param("_power", self.power)
            if isinstance(self.power, Callable)
            else self.power
        )
        self._amplitude = (
            self.param("_amplitude", self.amplitude)
            if isinstance(self.amplitude, Callable)
            else self.amplitude
        )

    def __call__(self) -> Field:
        return plane_wave(
            self.shape,
            self.dx,
            self.spectrum,
            self.spectral_density,
            self._power,
            self._amplitude,
            self._kykx,
            self.pupil,
            self.scalar,
        )


class GenericField(nn.Module):
    """
    Generates field with arbitrary ``phase`` and ``amplitude``.

    Can also be given ``pupil``.

    The attributes ``amplitude``, ``phase``, and ``power`` can be learned by
    using ``chromatix.utils.trainable``.

    Attributes:
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        amplitude: The amplitude of the field with shape `(B... H W C [1 | 3])`.
        phase: The phase of the field with shape `(B... H W C [1 | 3])`.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """

    dx: Union[float, Array]
    spectrum: Union[float, Array]
    spectral_density: Union[float, Array]
    amplitude: Union[Array, Callable[[PRNGKey], Array]]
    phase: Union[Array, Callable[[PRNGKey], Array]]
    power: Union[float, Callable[[PRNGKey], float]] = 1.0
    pupil: Optional[Callable[[Field], Field]] = None
    scalar: bool = True

    def setup(self):
        self._amplitude = (
            self.param("_amplitude", self.amplitude)
            if isinstance(self.amplitude, Callable)
            else self.amplitude
        )
        self._phase = (
            self.param("_phase", self.phase)
            if isinstance(self.phase, Callable)
            else self.phase
        )
        self._power = (
            self.param("_power", self.power)
            if isinstance(self.power, Callable)
            else self.power
        )

    def __call__(self) -> Field:
        return generic_field(
            self.dx,
            self.spectrum,
            self.spectral_density,
            self._amplitude,
            self._phase,
            self._power,
            self.pupil,
            self.scalar,
        )
