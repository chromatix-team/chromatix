from __future__ import annotations
import jax.numpy as jnp
from jax import vmap
from jax.lax import psum
from flax import linen as nn
from chex import Array, PRNGKey
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Union
from .field import Field
from .elements import FFLens, ObjectivePointSource, PhaseMask
from .ops import (
    fourier_convolution,
    shot_noise,
    approximate_shot_noise,
    init_plane_resample,
)
from .utils import center_crop


class Microscope(nn.Module):
    """
    Microscope with a planewise spatially invariant point spread function.

    This ``Microscope`` is a ``flax`` ``Module`` that accepts a function or
    ``Module`` that computes the point spread function (PSF) of the microscope.
    ``Microscope`` then uses this PSF to simulate imaging via a convolution
    of the sample with the specified PSF. Optionally, the sensor can also
    simulate noise.

    Attributes:
        system_psf: A function or ``Module`` that will compute the ``Field``
            just before the sensor plane due to a point source for this imaging
            system. Must take a ``Microscope`` as the first argument to read
            any relevant optical properties of the system. Can take any other
            arguments passed during a call to this ``Microscope`` (e.g. z
            values to compute a 3D PSF at for imaging).
        f: Focal length of the system's objective.
        n: Refractive index of the system's objective.
        NA: The numerical aperture of the system's objective. By
            default, no pupil is applied to the incoming ``Field``.
        spectrum: The wavelengths included in the simulation of the system's
            PSF.
        spectral_density: The weights of each wavelength in the simulation of
            the system's PSF.
        shot_noise_mode: A string of either 'approximate' or 'poisson' that
            determines how to add noise to the image. Defaults to None, in
            which case no noise is applied.
        psf_resampling_method: A string of either 'linear' or 'cubic' that
            determines how the PSF is resampled to the shape of the sensor.
        reduce_axis: If provided, the input will be reduced along this
            dimension.
        reduce_parallel_axis_name: If provided, psum along the axis with this
            name.
    """

    system_psf: Callable[[Microscope], Field]
    sensor_shape: Tuple[int, ...]
    sensor_spacing: float
    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Union[float, Callable[[PRNGKey], float]]
    spectrum: Array
    spectral_density: Array
    shot_noise_mode: Optional[Literal["approximate", "poisson"]] = None
    psf_resampling_method: Optional[Literal["pool", "linear", "cubic"]] = None
    reduce_axis: Optional[int] = None
    reduce_parallel_axis_name: Optional[str] = None

    def setup(self):
        self._f = self.param("f", self.f) if isinstance(self.f, Callable) else self.f
        self._n = self.param("n", self.n) if isinstance(self.n, Callable) else self.n
        self._NA = (
            self.param("NA", self.NA) if isinstance(self.NA, Callable) else self.NA
        )
        if self.psf_resampling_method is not None:
            self.resample = init_plane_resample(
                (*self.sensor_shape, 1), self.sensor_spacing, self.psf_resampling_method
            )

    def __call__(self, sample: Array, *args: Any, **kwargs: Any) -> Array:
        """
        Computes PSF and convolves PSF with ``data`` to simulate imaging.

        Args:
            sample: The sample to be imaged of shape `[B H W 1]`.
            *args: Any positional arguments needed for the PSF model.
            **kwargs: Any keyword arguments needed for the PSF model.
        """
        if self.psf_resampling_method is not None:
            psf = self.psf(*args, **kwargs)
            psf = vmap(self.resample, in_axes=(0, None))(
                psf.intensity, psf.dx[..., 0].squeeze()
            )
        else:
            psf = self.psf(*args, **kwargs).intensity
        return self.image(sample, psf)

    def psf(self, *args: Any, **kwargs: Any) -> Field:
        """Computes PSF complex field, taking any necessary arguments."""
        return self.system_psf(self, *args, **kwargs)

    def image(self, sample: Array, psf: Array) -> Array:
        """
        Computes image or batch of images using the specified PSF and sample.

        Potentially, this sensor function is a ``Module`` that can declare a
        `flax` RNG stream in order to simulate noise (e.g. shot noise), in
        which case a `flax` RNG stream is created with key "noise."

        Args:
            sample: The sample volume to image with, has shape `[B H W 1]`.
            psf: The PSF intensity volume to image with, has shape `[B H W 1]`.
        """
        image = vmap(fourier_convolution, in_axes=(0, 0))(sample, psf)
        if self.reduce_axis is not None:
            image = jnp.sum(image, axis=self.reduce_axis, keepdims=True)
        if self.reduce_parallel_axis_name is not None:
            image = psum(image, axis_name=self.reduce_parallel_axis_name)
        if self.shot_noise_mode is not None:
            noise_key = self.make_rng("noise")
            if self.shot_noise_mode == "approximate":
                image = approximate_shot_noise(noise_key, image)
            elif self.shot_noise_mode == "poisson":
                image = shot_noise(noise_key, image)
        return image


class OpticalSystem(nn.Module):
    """
    Combines a sequence of optical elements into a single ``Module``.

    Takes a sequence of functions or ``Module``s (any ``Callable``) and calls
    them in sequence, assuming each element of the sequence only accepts a
    ``Field`` as input and returns a ``Field`` as output, with the exception of
    the first element of the sequence, which can take any arguments necessary
    (e.g. to allow an element from ``chromatix.elements.sources`` to initialize
    a ``Field``) and the last element of the sequence, which may return an
    ``Array``. This is intended to mirror the style of deep learning libraries
    that describe a neural network as a sequence of layers, allowing for an
    optical system to be described conveniently as a list of elements.

    Attributes:
        elements: A sequence of optical elements describing the system.
    """

    elements: Sequence[Callable]

    @nn.compact
    def __call__(self, *args: Any, **kwargs: Any) -> Union[Field, Array]:
        """Returns the result of calling all elements in sequence."""
        field = self.elements[0](*args, **kwargs)  # allow field to be initialized
        for element in self.elements[1:]:
            field = element(field)
        return field


class Optical4FSystemPSF(nn.Module):
    shape: Tuple[int, int]
    spacing: float
    padding_ratio: float
    phase: Union[Array, Callable[[PRNGKey, Tuple[int, ...]], Array]]

    @nn.compact
    def __call__(self, microscope: Microscope, z: Array) -> Field:
        padding = tuple(int(s * self.padding_ratio) for s in self.shape)
        padded_shape = tuple(s + p for s, p in zip(self.shape, padding))
        required_spacing = self.compute_required_spacing(
            padded_shape[0],
            self.spacing,
            microscope.f,
            microscope.n,
            jnp.atleast_1d(microscope.spectrum),
        )
        system = OpticalSystem(
            [
                ObjectivePointSource(
                    padded_shape,
                    required_spacing,
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
        psf = system(z)
        psf = psf.replace(
            u=center_crop(psf.u, (0, padding[0] // 2, padding[1] // 2, 0))
        )
        return psf

    @staticmethod
    def compute_required_spacing(
        height: int, output_spacing: float, f: float, n: float, wavelength: float
    ) -> float:
        return f * wavelength / (n * height * output_spacing)
