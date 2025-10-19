from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, ScalarLike

from chromatix import Field, Sensor, Spectrum
from chromatix.functional import ff_lens, objective_point_source, phase_change
from chromatix.typing import wv
from chromatix.utils import sigmoid_taper

from ..ops import fourier_convolution
from ..utils import center_crop

__all__ = ["Microscope", "Optical4FSystemPSF"]


class Microscope(eqx.Module):
    """
    Microscope with a planewise spatially invariant point spread function.

    This ``Microscope`` is an ``equinox`` ``Module`` that accepts a function or
    ``Module`` that computes the point spread function (PSF) of the microscope.
    ``Microscope`` then uses this PSF to simulate imaging via a convolution of
    the provided sample with the specified PSF. This is useful for simulating
    images in the fully incoherent case where coherent propagation through
    samples is not an appropriate model, e.g. fluorescence imaging. Microscope
    also takes a [``Sensor``][chromatix.elements.sensors.BasicSensor] which can
    perform resampling of the PSF simulation to the desired sensor pixel pitch
    or simulate noise. Parameters of the provided PSF can be differentiated
    through, e.g. the phase mask of the Optical4FSystemPSF.

    Attributes:
        system_psf: A function or ``Module`` that will compute the field at
            the sensor plane due to a point source for this imaging system.
            Must take a ``Microscope`` as the first argument to read any
            relevant optical properties of the system. Can take any other
            arguments passed during a call to this ``Microscope`` (e.g. z
            values to compute a 3D PSF at for imaging). Can either return a
            ``Field``, in which case the intensity and spacing of the PSF will
            be automatically determined, or an ``Array``, in which case the
            ``Array`` is assumed to be the intensity and the spacing of the PSF
            will be automatically determined based on the ratio of the specified
            ``sensor_shape`` to the shape of the PSF. When the input is a
            ``Field``, the spacing is assumed to be equal for all wavelengths
            of the ``spectrum`` of the ``Field`` and the spacing for the first
            wavelength is used to calculate the resampling.
        sensor: The sensor used for imaging the sample. Must be an instance
            of [``Sensor``][chromatix.core.base.Sensor], e.g.
            [``BasicSensor``][chromatix.elements.sensors.BasicSensor]. This
            sensor may be used to perform resampling of the simulated PSF to the
            pixel pitch of the sensor.
        f: Focal length of the system's objective.
        n: Refractive index of the system's objective.
        NA: The numerical aperture of the system's objective.
        spectrum: The
            [``Spectrum``](core.md#chromatix.core.spectrum.Spectrum.build)
            of the ``Field`` to be created. This can be specified either as a
            single float value representing a wavelength in units of distance
            for a monochromatic field, a 1D array of wavelengths for a chromatic
            field that has the same intensity in all wavelengths, or a tuple
            of two 1D arrays where the first array represents the wavelengths
            and the second array is a unitless array of weights that define the
            spectral density (the relative intensity of each wavelength in the
            spectrum). This second array of spectral density will automatically
            be normalized to sum to 1.
        padding_ratio: The proportion of the original PSF shape that will be
            added to simulate the PSF. That means the final PSF simulation
            shape will be shape * (1.0 + padding) in each dimension. This will
            then automatically be cropped to the original desired shape after
            simulation. Defaults to 0, in which case no cropping occurs.
        taper_width: The width in pixels of the sigmoid that will be used to
            smoothly bring the edges of the PSF to 0. This helps to prevent
            edge artifacts in the image if the PSF has edge artifacts. Defaults
            to 0, in which case no tapering is applied.
        convolution_axes: The axes over which to perform convolution between the
            PSF and the sample. The PSF and the sample must have the same number
            of dimensions. Defaults to ``(-2, -1)`` for 2D convolution over the
            last two axes. If the PSF and the sample are 3D, this will convolve
            each plane of the PSF with each corresponding plane of the sample.
            If 3D convolution is desired instead (e.g. to simulate imaging a
            3D stack), you likely want to set this argument to ``(-3, -2, -1)``
            which will perform 3D convolution on the 3 axes of a 3D PSF and
            sample.
        fast_fft_shape: If `True`, Fourier convolutions will be computed at
            potentially larger shapes to gain speed at the expense of increased
            memory requirements. If you are running out of memory, try setting
            this to `False`. Defaults to `True`.
    """

    system_psf: Callable[[Microscope], Field | Array]
    sensor: Sensor
    f: ScalarLike = eqx.field(static=True)
    n: ScalarLike = eqx.field(static=True)
    NA: ScalarLike = eqx.field(static=True)
    spectrum: Spectrum
    padding_ratio: ScalarLike = eqx.field(static=True)
    taper_width: ScalarLike = eqx.field(static=True)
    convolution_axes: tuple[int, ...] = eqx.field(static=True)
    fast_fft_shape: bool = eqx.field(static=True)

    def __init__(
        self,
        system_psf: Callable[[Microscope, ...], Field | Array],
        sensor: Sensor,
        f: ScalarLike,
        n: ScalarLike,
        NA: ScalarLike,
        spectrum: Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]],
        padding_ratio: ScalarLike = 0,
        taper_width: ScalarLike = 0,
        convolution_axes: tuple[int, ...] = (-2, -1),
        fast_fft_shape: bool = True,
    ):
        """
        Args:
            system_psf: A function or ``Module`` that will compute the field at
                the sensor plane due to a point source for this imaging system.
                Must take a ``Microscope`` as the first argument to read any
                relevant optical properties of the system. Can take any other
                arguments passed during a call to this ``Microscope`` (e.g. z
                values to compute a 3D PSF at for imaging). Can either return
                a ``Field``, in which case the intensity and spacing of the PSF
                will be automatically determined, or an ``Array``, in which case
                the ``Array`` is assumed to be the intensity and the spacing of
                the PSF will be automatically determined based on the ratio of
                the specified ``sensor_shape`` to the shape of the PSF. When the
                input is a ``Field``, the spacing is assumed to be equal for all
                wavelengths of the ``spectrum`` of the ``Field`` and the spacing
                for the first wavelength is used to calculate the resampling.
            sensor: The sensor used for imaging the sample. Must be an
                instance of [``Sensor``][chromatix.core.base.Sensor], e.g.
                [``BasicSensor``][chromatix.elements.sensors.BasicSensor]. This
                sensor may be used to perform resampling of the simulated PSF to
                the pixel pitch of the sensor.
            f: Focal length of the system's objective.
            n: Refractive index of the system's objective.
            NA: The numerical aperture of the system's objective.
            spectrum: The
                [``Spectrum``](core.md#chromatix.core.spectrum.Spectrum.build)
                of the ``Field`` to be created. This can be specified either
                as a single float value representing a wavelength in units of
                distance for a monochromatic field, a 1D array of wavelengths
                for a chromatic field that has the same intensity in all
                wavelengths, or a tuple of two 1D arrays where the first array
                represents the wavelengths and the second array is a unitless
                array of weights that define the spectral density (the relative
                intensity of each wavelength in the spectrum). This second array
                of spectral density will automatically be normalized to sum
                to 1.
            padding_ratio: The proportion of the original PSF shape that will
                be added to simulate the PSF. That means the final shape will
                be shape * (1.0 + padding) in each dimension. This will then
                automatically be cropped to the original desired shape after
                simulation. Defaults to 0, in which case no cropping occurs.
            taper_width: The width in pixels of the sigmoid that will be used to
                smoothly bring the edges of the PSF to 0. This helps to prevent
                edge artifacts in the image if the PSF has edge artifacts.
                Defaults to 0, in which case no tapering is applied.
            convolution_axes: The axes over which to perform convolution between
                the PSF and the sample. The PSF and the sample must have the
                same number of dimensions. Defaults to ``(-2, -1)`` for 2D
                convolution over the last two axes. If the PSF and the sample
                are 3D, this will convolve each plane of the PSF with each
                corresponding plane of the sample. If 3D convolution is desired
                instead (e.g. to simulate imaging a 3D stack), you likely want
                to set this argument to ``(-3, -2, -1)`` which will perform 3D
                convolution on the 3 axes of a 3D PSF and sample.
            fast_fft_shape: If `True`, Fourier convolutions will be computed
                at potentially larger shapes to gain speed at the expense of
                increased memory requirements. If you are running out of memory,
                try setting this to `False`. Defaults to `True`.
        """
        self.system_psf = system_psf
        self.sensor = sensor
        self.f = f
        self.n = n
        self.NA = NA
        self.spectrum = Spectrum.build(spectrum)
        self.padding_ratio = padding_ratio
        self.taper_width = taper_width
        self.convolution_axes = convolution_axes
        self.fast_fft_shape = fast_fft_shape

    def __call__(
        self,
        sample: ArrayLike,
        *args: Any,
        key: PRNGKeyArray | None = None,
        **kwargs: Any,
    ) -> Array:
        """
        Computes PSF and convolves PSF with ``data`` to simulate imaging.

        Args:
            sample: The sample to be imaged of shape `(... height width)` for 2D
                samples or `(... depth height width)` for 3D samples.
            *args: Any positional arguments needed for the PSF model.
            **kwargs: Any keyword arguments needed for the PSF model.
        """
        system_psf = self.psf(*args, **kwargs)
        ndim = system_psf.ndim
        # NOTE(dd): We have to manually specify the spatial dimensions here
        # because we can have system_psf functions return Arrays in addition
        # to Fields.
        if isinstance(system_psf, Field):
            spatial_dims = tuple(ndim + s for s in system_psf.spatial_dims)
        else:
            # NOTE(dd): The system PSF is allowed to be an array. In this case,
            # we assume that this is an intensity array which has been cropped
            # of all trailing dimensions after calculation of the intensity
            # (i.e. that the height and width are the last two dimensions of
            # the array).
            spatial_dims = (ndim - 2, ndim - 1)
        psf = self._process_psf(system_psf, ndim, spatial_dims)
        return self.image(sample, psf, key=key)

    def psf(self, *args: Any, **kwargs: Any) -> Field | Array:
        """Computes PSF of system, taking any necessary arguments."""
        return self.system_psf(self, *args, **kwargs)

    def _process_psf(
        self,
        system_psf: Field | Array,
        ndim: int,
        spatial_dims: tuple[int, int],
    ) -> Array:
        """
        Prepare PSF to be convolved with a sample by doing the following:

        1. If the PSF was provided as a ``Field``, compute the intensity PSF
        2. Crop the PSF if it was padded
        3. Taper the edges of the PSF with the specified ``taper_width``
        4. Resample the PSF to the shape of the specified ``sensor``
        """
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
            spacing = system_psf.central_rectangular_dx
        else:
            psf = system_psf
            spacing = self.sensor.spacing * jnp.array(
                [
                    self.sensor.shape[0] / (shape[0] - padding[0]),
                    self.sensor.shape[1] / (shape[1] - padding[1]),
                ]
            )
        if self.padding_ratio > 0:
            pad_spec: list[Any] = [None for _ in range(ndim)]
            pad_spec[spatial_dims[0]] = padding[0] // 2
            pad_spec[spatial_dims[1]] = padding[1] // 2
            psf = center_crop(psf, pad_spec)
        if self.taper_width > 0:
            psf = psf * sigmoid_taper(unpadded_shape, self.taper_width)  # type: ignore
        psf = self.sensor.resample(psf, spacing)
        return psf

    def image(
        self,
        sample: ArrayLike,
        psf: ArrayLike,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        """
        Computes image or batch of images using the specified ``psf`` and
        ``sample``. Assumes that both the ``sample`` and ``psf`` have already
        been sampled to the pixels of the sensor.

        Args:
            sample: The sample volume to image with of shape `(... height
                width)` for 2D samples or `(... depth height width)` for 3D
                samples.
            psf: The PSF intensity volume to image with of shape `(... height
                width)` for 2D samples or `(... depth height width)` for 3D
                samples.
        """
        image = fourier_convolution(
            sample, psf, axes=self.convolution_axes, fast_fft_shape=self.fast_fft_shape
        )
        # NOTE(dd): By this point, the image should already be at the same
        # spacing as the sensor. Any resampling to the pixels of the sensor
        # should already have happened to the PSF.
        image = self.sensor(image, self.sensor.spacing, resample=False, key=key)
        return image


class Optical4FSystemPSF(eqx.Module):
    """
    Simulates the point spread function (PSF) of a 4f system with a phase mask.

    Attributes:
        shape: A tuple of form `(height width)` defining the number of pixels
            used to simulate the PSF at each plane.
        spacing: The desired output spacing of the PSF once it is simulated,
            e.g. the camera sensor pixel pitch.
        f_tube: The focal length of the tube lens of the system in units of
            distance. The focal length of the objective lens will be accessed
            through the ``Microscope`` passed during a call to this ``Module``.
        phase: The phase mask for the 4f simulation. Must be a 2D array of
            phase values in units of radians with shape `(height width)` where
            `(height width)` match the shape of the simulation.
    """

    shape: tuple[int, int] = eqx.field(static=True)
    spacing: ScalarLike = eqx.field(static=True)
    f_tube: ScalarLike = eqx.field(static=True)
    phase: ArrayLike

    def __init__(
        self,
        shape: tuple[int, int],
        spacing: ScalarLike,
        f_tube: ScalarLike,
        phase: ArrayLike,
    ):
        self.shape = shape
        self.spacing = spacing
        self.f_tube = f_tube
        self.phase = phase

    def __call__(self, microscope: Microscope, z: ArrayLike) -> Field:
        padding = tuple(int(s * microscope.padding_ratio) for s in self.shape)
        padded_shape = tuple(s + p for s, p in zip(self.shape, padding))
        required_spacing = self.compute_required_spacing(
            padded_shape[0],
            self.spacing,
            self.f_tube,
            microscope.n,
            microscope.spectrum.wavelength,
        )
        field = objective_point_source(
            padded_shape,
            required_spacing,
            microscope.spectrum,
            z,
            microscope.f,
            microscope.n,
            microscope.NA,
        )
        field = phase_change(field, self.phase)
        field = ff_lens(field, self.f_tube, microscope.n)
        return field

    @staticmethod
    def compute_required_spacing(
        height: int,
        output_spacing: ScalarLike,
        f: ScalarLike,
        n: ScalarLike,
        spectrum: ScalarLike,
    ) -> ScalarLike:
        return f * spectrum / (n * height * output_spacing)
