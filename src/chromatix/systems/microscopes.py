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

    This ``Microscope`` is a ``flax`` ``Module`` that accepts a function
    or ``Module`` that computes the point spread function (PSF) of the
    microscope. ``Microscope`` then uses this PSF to simulate imaging via a
    convolution of the sample with the specified PSF. Optionally, the sensor
    can also simulate noise. Parameters of the provided PSF are allowed to be
    trainable (using ``chromatix.utils.trainable``), e.g. the phase mask of the
    Optical4FSystemPSF.

    Attributes:
        system_psf: A function or ``Module`` that will compute the field at the
            sensor plane due to a point source for this imaging system. Must
            take a ``Microscope`` as the first argument to read any relevant
            optical properties of the system. Can take any other arguments
            passed during a call to this ``Microscope`` (e.g. z values to
            compute a 3D PSF at for imaging). Can either return a ``Field``, in
            which case the intensity and spacing of the PSF will automatically
            be determined, or an ``Array``, in which case the ``Array`` is
            assumed to be the intensity and the spacing of the PSF will be
            automatically determined based on the ratio of the specified
            ``sensor_shape`` to the shape of the PSF.
        sensor_shape: A tuple of form (H W) defining the camera sensor shape.
        sensor_spacing: A float defining the pixel pitch of the camera sensor.
        f: Focal length of the system's objective.
        n: Refractive index of the system's objective.
        NA: The numerical aperture of the system's objective.
        spectrum: The wavelengths included in the simulation of the system's
            PSF.
        spectral_density: The weights of each wavelength in the simulation of
            the system's PSF.
        padding_ratio: The proportion of the original PSF shape that will be
            added to simulate the PSF. That means the final shape will be shape
            * (1.0 + padding) in each dimension. This will then automatically
            be cropped to the original desired shape after simulation. Defaults
            to 0, in which case no cropping occurs.
        taper_width: The width in pixels of the sigmoid that will be used to
            smoothly bring the edges of the PSF to 0. This helps to prevent
            edge artifacts in the image if the PSF has edge artifacts. Defaults
            to 0, in which case no tapering is applied.
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

    system_psf: Callable[[Microscope], Union[Field, Array]]
    sensor_shape: Tuple[int, int]
    sensor_spacing: float
    f: float
    n: float
    NA: float
    spectrum: Array
    spectral_density: Array
    padding_ratio: float = 0
    taper_width: float = 0
    shot_noise_mode: Optional[Literal["approximate", "poisson"]] = None
    psf_resampling_method: Optional[Literal["pool", "linear", "cubic"]] = None
    reduce_axis: Optional[int] = None
    reduce_parallel_axis_name: Optional[str] = None

    def setup(self):
        if self.psf_resampling_method is not None:
            self.resample = init_plane_resample(
                (*self.sensor_shape, 1), self.sensor_spacing, self.psf_resampling_method
            )

    def __call__(self, sample: Array, *args: Any, **kwargs: Any) -> Array:
        """
        Computes PSF and convolves PSF with ``data`` to simulate imaging.

        Args:
            sample: The sample to be imaged of shape `(B H W 1)`.
            *args: Any positional arguments needed for the PSF model.
            **kwargs: Any keyword arguments needed for the PSF model.
        """
        system_psf = self.psf(*args, **kwargs)
        ndim = system_psf.ndim
        # NOTE(dd): We have to manually calculate the spatial dimensions here
        # because we can have system_psf functions return Arrays in addition
        # to Fields.
        spatial_dims = (1 + ndim - 4, 2 + ndim - 4)
        psf = self._process_psf(system_psf, ndim, spatial_dims)
        return self.image(sample, psf, axes=spatial_dims)

    def psf(self, *args: Any, **kwargs: Any) -> Union[Field, Array]:
        """Computes PSF of system, taking any necessary arguments."""
        return self.system_psf(self, *args, **kwargs)

    def _process_psf(
        self, system_psf: Union[Field, Array], ndim: int, spatial_dims: Tuple[int, int]
    ) -> Array:
        shape = system_psf.shape[spatial_dims[0] : spatial_dims[1] + 1]
        if self.padding_ratio is not None:
            unpadded_shape = tuple(int(s / (1.0 + self.padding_ratio)) for s in shape)
            padding = tuple(s - u for s, u in zip(shape, unpadded_shape))
        else:
            unpadded_shape = shape
            padding = (0, 0)
        # WARNING(dd): Assumes that field has same spacing at all wavelengths
        # when calculating intensity!
        if isinstance(system_psf, Field):
            psf = system_psf.intensity
            spacing = system_psf.dx[..., 0].squeeze()
        else:
            psf = system_psf
            spacing = self.sensor_spacing * (
                self.sensor_shape[0] / (psf.shape[1] - padding[0])
            )
        if self.padding_ratio > 0:
            pad_spec = [None for _ in range(ndim)]
            pad_spec[spatial_dims[0]] = padding[0] // 2
            pad_spec[spatial_dims[1]] = padding[1] // 2
            psf = center_crop(psf, pad_spec)
        if self.taper_width > 0:
            psf = psf * sigmoid_taper(unpadded_shape, self.taper_width, ndim=ndim)
        if self.psf_resampling_method is not None:
            for i in range(ndim - 3):
                resample = vmap(self.resample, in_axes=(0, None))
            psf = resample(psf, spacing)
        return psf

    def image(self, sample: Array, psf: Array, axes: Tuple[int, int] = (1, 2)) -> Array:
        """
        Computes image or batch of images using the specified PSF and sample.

        Args:
            sample: The sample volume to image with, has shape `(B... H W 1)`.
            psf: The PSF intensity volume to image with, has shape `(B... H W 1)`.
        """
        image = fourier_convolution(psf, sample, axes=axes)
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
            simulate the PSF at each plane.
        spacing: The desired output spacing of the PSF once it is simulated,
            i.e. the spacing at the camera plane when the PSF is measured. Note
            that this **does not** need to match the actual spacing of the
            sensor, and often should be a finer spacing than the camera.
        phase: The phase mask for the 4f simulation. Must be an array of
            shape (H W) where (H W) match the shape of the simulation, or an
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
