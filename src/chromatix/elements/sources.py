import jax.numpy as jnp
import flax.linen as nn
from ..field import Field
from ..functional.sources import (
    plane_wave,
    point_source,
    objective_point_source,
    generic_field,
)
from typing import Optional, Callable, Tuple, Union
from chex import PRNGKey, Array
from chromatix.elements.utils import register

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
            the polarization.
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

    @nn.compact
    def __call__(self) -> Field:
        power = register(self, "power")
        z = register(self, "z")
        n = register(self, "n")
        amplitude = register(self, "amplitude")
        return point_source(
            self.shape,
            self.dx,
            self.spectrum,
            self.spectral_density,
            z,
            n,
            power,
            amplitude,
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
            the polarization.
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

    @nn.compact
    def __call__(self, z: float) -> Field:
        f = register(self, "f")
        n = register(self, "n")
        NA = register(self, "NA")
        power = register(self, "power")
        amplitude = register(self, "amplitude")

        return objective_point_source(
            self.shape,
            self.dx,
            self.spectrum,
            self.spectral_density,
            z,
            f,
            n,
            NA,
            power,
            amplitude,
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
            the polarization.
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
    kykx: Union[Array, Tuple[float, float]] = (0.0, 0.0)
    pupil: Optional[Callable[[Field], Field]] = None
    scalar: bool = True

    @nn.compact
    def __call__(self) -> Field:
        kykx = register(self, "kykx")
        power = register(self, "power")
        amplitude = register(self, "amplitude")
        return plane_wave(
            self.shape,
            self.dx,
            self.spectrum,
            self.spectral_density,
            power,
            amplitude,
            kykx,
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

    @nn.compact
    def __call__(self) -> Field:
        amplitude = register(self, "amplitude")
        phase = register(self, "phase")
        power = register(self, "power")

        return generic_field(
            self.dx,
            self.spectrum,
            self.spectral_density,
            amplitude,
            phase,
            power,
            self.pupil,
            self.scalar,
        )
