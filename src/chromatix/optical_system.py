from __future__ import annotations
from flax import linen as nn
from chex import Array, PRNGKey
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from .field import Field
from .elements import FFLens, ObjectivePointSource, PhaseMask


class Microscope(nn.Module):
    """
    Microscope with a point spread function (spatially invariant in each plane).

    This ``Microscope`` is a ``flax`` ``Module`` that accepts a function or
    ``Module`` that computes the point spread function (PSF) of the microscope.
    ``Microscope`` then uses this PSF to simulate imaging using a function or
    ``Module`` that defines a sensor. The reason that this thin wrapper exists
    is to enable fast simulations of cases where a system has a single PSF (as
    opposed to a point response function or PRF that varies across the field).
    In these cases, the sensor function can perform a simple convolution of the
    sample with the specified PSF. Optionally, this sensor can also simulate
    noise.

    For example, if a batch represents planes of a volume in
    ``sample``, then setting ``sensor_fn`` as shown:

    ```python
    from chromatix.optical_system import Microscope
    from chromatix.elements import ShiftInvariantSensor
    microscope = Microscope(
        psf_spacing=...,
        n=...,
        f=...,
        NA=...,
        spectrum=...,
        spectral_density=...,
        psf_fn=...,
        sensor_fn=ShiftInvariantSensor(...))
    )
    ```

    will create a ``Microscope`` that convolves a depth-varying 3D PSF
    intensity with an input volume and sums the resulting planes to simulate
    taking an image of a 3D sample, where light from all planes arrives on
    the sensor.

    Further, to simulate noise (e.g. sensor read noise or shot noise) a
    ``noise_fn`` can be optionally provided, which will be called after the
    ``reduce_fn`` (if it has been provided). For example, continuing the
    example from above, we can simulate Poisson shot noise as shown:

    ```python
    from chromatix.optical_system import Microscope
    from chromatix.elements import ShiftInvariantSensor
    microscope = Microscope(
        psf_spacing=...,
        n=...,
        f=...,
        NA=...,
        spectrum=...,
        spectral_density=...,
        psf_fn=...,
        sensor_fn=ShiftInvariantSensor(..., shot_noise_mode='poisson'))
    )
    ```

    Attributes:
        system_psf: A function or ``Module`` that will compute the ``Field``
            just before the sensor plane due to a point source for this imaging
            system. Must take a ``Microscope`` as the first argument to read
            any relevant optical properties of the system. Can take any other
            arguments passed during a call to this ``Microscope`` (e.g. z
            values to compute a 3D PSF at for imaging).
        sensor: A function or ``Module`` that simulates an imaging sensor,
            producing a image (or batch of images) using the specified PSF and
            sample. Potentially, this sensor_fn also simulates noise (in which
            case a `flax` RNG stream is created with key "noise").
        f: Focal length of the objective.
        n: Refractive index of the objective.
        NA: The numerical aperture of the objective. By default, no pupil is
            applied to the incoming ``Field``.
        spectrum: The wavelengths included in the simulation.
        spectral_density: The weights of each wavelength in the simulation.
    """

    system_psf: Callable[[Microscope], Field]
    sensor: Callable[[Array, Array, Optional[PRNGKey]], Array]
    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Union[float, Callable[[PRNGKey], float]]
    spectrum: Array
    spectral_density: Array

    def setup(self):
        self._f = self.param("f", self.f) if isinstance(self.f, Callable) else self.f
        self._n = self.param("n", self.n) if isinstance(self.n, Callable) else self.n
        self._NA = (
            self.param("NA", self.NA) if isinstance(self.NA, Callable) else self.NA
        )

    def __call__(self, sample: Array, *args: Any, **kwargs: Any) -> Array:
        """
        Computes PSF and convolves PSF with ``data`` to simulate imaging.

        Args:
            data: The sample to be imaged of shape `[B H W C]`.
            *args: Any positional arguments needed for the PSF model.
            **kwargs: Any keyword arguments needed for the PSF model.
        """
        psf = self.psf_intensity(*args, **kwargs)
        return self.image(sample, psf)

    def psf_field(self, *args: Any, **kwargs: Any) -> Field:
        """Computes PSF complex field, taking any necessary arguments."""
        return self.system_psf(self, *args, **kwargs)

    def psf_intensity(self, *args: Any, **kwargs: Any) -> Array:
        """Computes PSF intensity, taking any necessary arguments."""
        return self.system_psf(self, *args, **kwargs).intensity

    def image(self, sample: Array, psf: Array) -> Array:
        """
        Computes image or batch of images using the specified PSF and sample.

        Potentially, this sensor function is a ``Module`` that can declare a
        `flax` RNG stream in order to simulate noise (e.g. shot noise), in
        which case a `flax` RNG stream is created with key "noise."

        Args:
            psf: The PSF intensity volume to image with, has shape `[B H W C]`.
            data: The sample volume to image with, has shape `[B H W C]`.
        """
        return self.sensor(sample, psf)


class OpticalSystem(nn.Module):
    """
    Combines a sequence of optical elements into a single ``Module``.

    Takes a sequence of functions or ``Module``s (any ``Callable``) and calls
    them in sequence, assuming each element of the sequence only accepts a
    ``Field`` as input and returns a ``Field`` as output, with the exception of
    the first element of the sequence, which can take any arguments necessary
    (e.g. to allow an element from ``chromatix.elements.sources`` to initialize
    a ``Field``). This is intended to mirror the style of deep learning
    libraries that describe a neural network as a sequence of layers, allowing
    for an optical system to be described conveniently as a list of elements.

    Attributes:
        elements: A sequence of optical elements describing the system.
    """

    elements: Sequence[Callable]

    @nn.compact
    def __call__(self, *args: Any, **kwargs: Any) -> Field:
        """Returns a ``Field`` by calling all elements in sequence."""
        field = self.elements[0](*args, **kwargs)  # allow field to be initialized
        for element in self.elements[1:]:
            field = element(field)
        return field


class Optical4FSystemPSF(nn.Module):
    shape: Tuple[int, int]
    spacing: float
    downsample: int
    phase: Union[Array, Callable[[PRNGKey, Tuple[int, ...]], Array]]

    @nn.compact
    def __call__(self, microscope: Microscope, z: Array) -> Field:
        system = OpticalSystem(
            [
                ObjectivePointSource(
                    self.shape,
                    self.spacing,
                    microscope.spectrum,
                    microscope.spectral_density,
                    microscope.f,
                    microscope.n,
                    microscope.NA,
                ),
                PhaseMask(self.phase),
                FFLens(microscope.f, microscope.n, microscope.NA),
            ]
        )
        return system(z)
