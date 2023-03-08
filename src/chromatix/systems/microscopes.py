from __future__ import annotations
import jax.numpy as jnp
from jax import vmap
from jax.lax import psum
from flax import linen as nn
from chex import Array, PRNGKey
from typing import Any, Callable, Literal, Optional, Tuple, Union
from ..field import Field
from ..elements import FFLens, ObjectivePointSource, PhaseMask
from ..ops import (
    fourier_convolution,
    shot_noise,
    sigmoid_taper,
    approximate_shot_noise,
    init_plane_resample,
)
from ..utils import center_crop
from .optical_system import OpticalSystem

__all__ = ["Microscope", "Optical4FSystemPSF"]


class Microscope(nn.Module):
    """
    Microscope with a planewise spatially invariant point spread function.

    This ``Microscope`` is a ``flax`` ``Module`` that accepts a function or
    ``Module`` that computes the point spread function (PSF) of the microscope.
    ``Microscope`` then uses this PSF to simulate imaging via a convolution of
    the sample with the specified PSF. Optionally, the sensor can also simulate
    noise. Parameters of the provided PSF are allowed to be trainable (using
    ``chromatix.utils.trainable``). Further, this system's objective focal
    length, objective refractive index, and objective numerical aperture can
    also be trainable.

    Attributes:
        system_psf: A function or ``Module`` that will compute the ``Field``
            just before the sensor plane due to a point source for this imaging
            system. Must take a ``Microscope`` as the first argument to read
            any relevant optical properties of the system. Can take any other
            arguments passed during a call to this ``Microscope`` (e.g. z
            values to compute a 3D PSF at for imaging).
        padding_ratio: The proportion of the original PSF shape that will be
            added to simulate the PSF. That means the final shape will be shape
            * (1.0 + padding) in each dimension. This will then automatically
            be cropped to the original desired shape after simulation, and a
            taper will be applied to the result.
        taper_width: The width in pixels of the sigmoid that will be used to
            smoothly bring the edges of the PSF to 0. This helps to prevent
            edge artifacts in the image if the PSF has edge artifacts.
        sensor_shape: A tuple of form (H W) defining the camera sensor shape.
        sensor_spacing: A float defining the pixel pitch of the camera sensor.
        n: Refractive index of the system's objective.
        f: Focal length of the system's objective.
        NA: The numerical aperture of the system's objective.
        spectrum: The wavelengths included in the simulation of the system's
            PSF.
        spectral_density: The weights of each wavelength in the simulation of
            the system's PSF.
        shot_noise_mode: A string of either 'approximate' or 'poisson' that
            determines how to add noise to the image. Defaults to None, in
            which case no noise is applied.
        psf_resampling_method: A string of either 'linear' or 'cubic' that
            determines how the PSF is resampled to the shape of the sensor.
            Note that this assumes that the PSF that is computed has the same
            field of view as the sensor.
        reduce_axis: If provided, the input will be summed along this
            dimension.
        reduce_parallel_axis_name: If provided, psum along the axis with this
            name.
    """

    system_psf: Callable[[Microscope], Field]
    padding_ratio: float
    taper_width: float
    sensor_shape: Tuple[int, int]
    sensor_spacing: float
    n: Union[float, Callable[[PRNGKey], float]]
    f: Union[float, Callable[[PRNGKey], float]]
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
        psf = self.psf(*args, **kwargs)
        spacing = psf.dx[..., 0].squeeze()
        padding = tuple(int(self.padding_ratio * s) for s in self.system_psf.shape)
        psf = center_crop(psf.intensity, (None, padding[0] // 2, padding[1] // 2, None))
        psf = psf * sigmoid_taper(self.system_psf.shape, self.taper_width)
        if self.psf_resampling_method is not None:
            psf = vmap(self.resample, in_axes=(0, None))(psf, spacing)
        return self.image(sample, psf)

    def psf(self, *args: Any, **kwargs: Any) -> Field:
        """Computes PSF complex field, taking any necessary arguments."""
        return self.system_psf(self, *args, **kwargs)

    def image(self, sample: Array, psf: Array) -> Array:
        """
        Computes image or batch of images using the specified PSF and sample.

        Args:
            sample: The sample volume to image with, has shape `[B H W 1]`.
            psf: The PSF intensity volume to image with, has shape `[B H W 1]`.
        """
        image = vmap(fourier_convolution, in_axes=(0, 0))(psf, sample)
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


class Optical4FSystemPSF(nn.Module):
    """
    Simulates the point spread function (PSF) of a 4f system with a phase mask.

    The ``phase`` can be learned (pixel by pixel) by using
    ``chromatix.utils.trainable``.

    Attributes:
        shape: A tuple of form (H W) defining the number of pixels used to
            simulate the PSF.
        spacing: The desired output spacing of the PSF once it is simulated,
            i.e. the spacing at the camera plane when the PSF is measured. Note
            that this **does not** need to match the actual spacing of the
            sensor, and often should be a finer spacing than the camera.
        phase: The phase mask for the 4f simulation. Must be an array of shape
            (1 H W 1) where (H W) match the shape of the simulation, or an
            initialization function (e.g. using trainable).
    """

    shape: Tuple[int, int]
    spacing: float
    phase: Union[Array, Callable[[PRNGKey, Tuple[int, ...]], Array]]

    @nn.compact
    def __call__(self, microscope: Microscope, z: Array) -> Field:
        padding = tuple(int(s * microscope.padding_ratio) for s in self.shape)
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
                PhaseMask(self.phase, microscope.f, microscope.n, microscope.NA),
                FFLens(microscope.f, microscope.n, microscope.NA),
            ]
        )
        psf = system(z)
        return psf

    @staticmethod
    def compute_required_spacing(
        height: int, output_spacing: float, f: float, n: float, wavelength: Array
    ) -> float:
        return f * wavelength / (n * height * output_spacing)
