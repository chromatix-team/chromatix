import flax.linen as nn
from chromatix import Field
from ..functional.sources import (
    empty_field,
    plane_wave,
    point_source,
    objective_point_source,
    generic_field,
)

from typing import Optional, Callable
from chex import PRNGKey, Array

__all__ = ["PointSource", "ObjectivePointSource", "PlaneWave", "GenericBeam"]


class PointSource(nn.Module):
    shape: tuple[int, int]
    dx: float
    spectrum: float
    spectral_density: float
    z: float | Callable[[PRNGKey], float]
    n: float | Callable[[PRNGKey], float]
    power: Optional[float | Callable[[PRNGKey], float]] = 1.0
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
    shape: tuple[int, int]
    dx: float
    spectrum: float
    spectral_density: float
    f: float | Callable[[PRNGKey], float]
    n: float | Callable[[PRNGKey], float]
    NA: float | Callable[[PRNGKey], float]
    power: Optional[float | Callable[[PRNGKey], float]] = 1.0

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
    shape: tuple[int, int]
    dx: float
    spectrum: float
    spectral_density: float
    power: Optional[float | Callable[[PRNGKey], float]] = 1.0
    phase: Optional[float | Callable[[PRNGKey], float]] = 0.0
    k_offset: Optional[Array | Callable[[PRNGKey], Array]] = None
    pupil: Optional[Callable[[Field], Field]] = None

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
        self._k_offset = (
            self.param("_NA", self.k_offset)
            if isinstance(self.k_offset, Callable)
            else self.k_offset
        )

    def __call__(self) -> Field:
        return plane_wave(
            self.empty_field, self._power, self._phase, self.pupil, self._k_offset
        )


class GenericBeam(nn.Module):
    shape: tuple[int, int]
    dx: float
    spectrum: float
    spectral_density: float
    amplitude: Array | Callable[[PRNGKey, tuple[int, int]], Array]
    phase: Array | Callable[[PRNGKey, tuple[int, int]], Array]
    power: Optional[float | Callable[[PRNGKey], float]] = 1.0
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

    def __call__(self) -> Field:
        return generic_field(
            self.empty_field,
            self._amplitude,
            self._phase,
            self._power,
            self.pupil,
        )
