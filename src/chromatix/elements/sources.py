import flax.linen as nn
from chromatix import Field
import jax.numpy as jnp
from ..utils.utils import get_wave_vector

from ..functional.sources import (
    empty_field,
    plane_wave,
    point_source,
    objective_point_source,
    generic_field,
    vector_plane_wave,
)

from typing import Optional, Callable, Tuple, Union
from chex import PRNGKey, Array

__all__ = [
    "PointSource",
    "ObjectivePointSource",
    "PlaneWave",
    "GenericBeam",
    "VectorPlaneWave",
]


class PointSource(nn.Module):
    """
    Generates field due to point source a distance ``z`` away.

    Can also be given ``pupil``.

    The attributes ``z``, ``n``, and ``power`` can be learned by using
    ``chromatix.utils.trainable``.

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
        pupil: If provided, will be called on the field to apply a pupil.
    """

    shape: Tuple[int, int]
    dx: float
    spectrum: float
    spectral_density: float
    z: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    power: Optional[Union[float, Callable[[PRNGKey], float]]] = 1.0
    pupil: Optional[Callable[[Field], Field]] = None

    def setup(self):
        self.empty_field = empty_field(
            self.shape, self.dx, self.spectrum, self.spectral_density
        )

        self._z = self.param("_z", self.z) if isinstance(self.z, Callable) else self.z
        self._n = self.param("_n", self.n) if isinstance(self.n, Callable) else self.n
        self._power = (
            self.param("_power", self.power)
            if isinstance(self.power, Callable)
            else self.power
        )

    def __call__(self) -> Field:
        return point_source(self.empty_field, self._z, self._n, self._power, self.pupil)


class ObjectivePointSource(nn.Module):
    """
    Generates field due to a point source defocused by an amount ``z`` away
    from the focal plane, just after passing through a lens with focal length
    ``f`` and numerical aperture ``NA``.

    The attributes ``f``, ``n``, ``NA``, and ``power`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        z: The distance of the point source.
        f: Focal length of the objective lens.
        n: Refractive index.
        NA: The numerical aperture of the objective lens.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
    """

    shape: Tuple[int, int]
    dx: float
    spectrum: float
    spectral_density: float
    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Union[float, Callable[[PRNGKey], float]]
    power: Optional[Union[float, Callable[[PRNGKey], float]]] = 1.0

    def setup(self):
        self.empty_field = empty_field(
            self.shape, self.dx, self.spectrum, self.spectral_density
        )

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

    def __call__(self, z: float) -> Field:
        return objective_point_source(
            self.empty_field, z, self._f, self._n, self._NA, self._power
        )


class PlaneWave(nn.Module):
    """
    Generates plane wave of given ``phase`` and ``power``.

    Can also be given ``pupil`` and ``k`` vector.

    The attributes ``power``, ``phase``, and ``k`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        phase: The phase of the plane wave in radians, defaults to 0.0.
        pupil: If provided, will be called on the field to apply a pupil.
        k: If provided, defines the orientation of the plane wave. Should be an
            array of shape `[2 H W]`. If provided, ``phase`` is ignored.
    """

    shape: Tuple[int, int]
    dx: float
    spectrum: float
    spectral_density: float
    power: Optional[Union[float, Callable[[PRNGKey], float]]] = 1.0
    phase: Optional[Union[float, Callable[[PRNGKey], float]]] = 0.0
    pupil: Optional[Callable[[Field], Field]] = None
    k: Optional[Union[Array, Callable[[PRNGKey], Array]]] = None

    def setup(self):
        self.empty_field = empty_field(
            self.shape, self.dx, self.spectrum, self.spectral_density
        )
        self._power = (
            self.param("_power", self.power)
            if isinstance(self.power, Callable)
            else self.power
        )
        self._phase = (
            self.param("_phase", self.phase)
            if isinstance(self.phase, Callable)
            else self.phase
        )
        self._k = self.param("_k", self.k) if isinstance(self.k, Callable) else self.k

    def __call__(self) -> Field:
        return plane_wave(
            self.empty_field, self._power, self._phase, self.pupil, self._k
        )


class GenericBeam(nn.Module):
    """
    Generates field with arbitrary ``phase`` and ``amplitude``.

    Can also be given ``pupil``.

    The attributes ``amplitude``, ``phase``, and ``power`` can be learned by
    using ``chromatix.utils.trainable``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        amplitude: The amplitude of the field with shape `[B H W C]`.
        phase: The phase of the field with shape `[B H W C]`.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        pupil: If provided, will be called on the field to apply a pupil.
    """

    shape: Tuple[int, int]
    dx: float
    spectrum: float
    spectral_density: float
    amplitude: Union[Array, Callable[[PRNGKey, Tuple[int, int]], Array]]
    phase: Union[Array, Callable[[PRNGKey, Tuple[int, int]], Array]]
    power: Optional[Union[float, Callable[[PRNGKey], float]]] = 1.0
    pupil: Optional[Callable[[Field], Field]] = None

    def setup(self):
        self.empty_field = empty_field(
            self.shape, self.dx, self.spectrum, self.spectral_density
        )

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
            self.empty_field,
            self._amplitude,
            self._phase,
            self._power,
            self.pupil,
        )


class VectorPlaneWave(nn.Module):
    """
    Generates plane wave of given ``phase`` and ``power``.

    Can also be given ``pupil`` and ``k`` vector in a ky, kx order.

    The attributes ``power``, ``phase``, and ``k`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        phase: The phase of the plane wave in radians, defaults to 0.0.
        pupil: If provided, will be called on the field to apply a pupil.
        k: If provided, defines the orientation of the plane wave. Should be an
            array of [ky, kx]. If provided, ``phase`` is ignored. The default one
            is [z, 0, 0]
        Ep: If provided, defines the initial polarization state of the polarization
            light. The default one is [0, 1, 1] which is a linear polarized light
    """

    shape: Tuple[int, int]
    dx: float
    n: float
    spectrum: float
    spectral_density: float
    k: Optional[Union[Array, Callable[[PRNGKey], float]]] = jnp.array([0.0, 0.0])
    Ep: Optional[Union[Array, Callable[[PRNGKey], float]]] = jnp.array([0, 1, 1])
    phase: Optional[Union[float, Callable[[PRNGKey], float]]] = 0.0
    pupil: Optional[Callable[[Field], Field]] = None

    def setup(self):
        self.empty_field = empty_field(
            self.shape, self.dx, self.spectrum, self.spectral_density, True
        )
        self._phase = (
            self.param("_phase", self.phase)
            if isinstance(self.phase, Callable)
            else self.phase
        )
        self._k = self.param("_k", self.k) if isinstance(self.k, Callable) else self.k
        self._Ep = (
            self.param("_Ep", self.Ep) if isinstance(self.Ep, Callable) else self.Ep
        )

        k = get_wave_vector(self.spectrum, self.k, self.n)
        ValuesEpk = jnp.dot(self._Ep, k)
        assert (
            ValuesEpk == 0
        ), "Isotropic media, the polarization vector should be orthogonal to the propagation vector."

    def __call__(self) -> Field:
        return vector_plane_wave(self.empty_field, self.k, self.Ep)
