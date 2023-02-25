from flax import linen as nn
from jax import vmap

from .ops import fourier_convolution

from .field import Field
from chex import Array, PRNGKey
from typing import Callable, Sequence, Optional, Any


class Microscope(nn.Module):
    """
    Microscope with a point spread function (spatially invariant in each plane).

    This ``Microscope`` is a ``flax`` ``Module`` that accepts a sequence of
    functions or ``Module``s which form an ``OpticalSystem``. This sequence of
    ``Callable``s are assumed to compute the point spread function (PSF) of the
    microscope. ``Microscope`` takes the intensity of this PSF and convolves it
    (respecting the batch dimension of the input) with the given ``data``.

    Optionally, ``reduce_fn`` can be provided which will be called on the
    resulting image. For example, if a batch represents planes of a volume in
    ``data``, then setting ``reduce_fn`` as shown:

    ```python
    from chromatix.optical_system import Microscope
    microscope = Microscope(
        optical_system=...,
        reduce_fn=lambda image: jnp.sum(image, axis=0)
    )
    ```

    will create a ``Microscope`` that convolves a depth-varying 3D PSF
    intensity with an input volume and sums the resulting planes to simulate
    taking an image of a 3D sample, where light from all planes arrives on
    the sensor.

    Further, to simulate noise (e.g. sensor read noise or shot noise) a
    ``noise_fn`` can be optionally provided, which will be called after the
    ``reduce_fn`` (if it has been provided). For example, continuing the
    example from above, we can simulate shot noise as shown:

    ```python
    from chromatix.optical_system import Microscope
    from chromatix.ops.noise import shot_noise
    microscope = Microscope(
        optical_system=...,
        noise_fn=shot_noise,
        reduce_fn=lambda image: jnp.sum(image, axis=0)
    )
    ```

    Attributes:
        optical_system: A sequence of functions or ``Module``s that will be
            used to construct an ``OpticalSystem`` that is assumed to output
            a ``Field``. The intensity of this ``Field`` will be interpreted
            as the PSF used for imaging.
        reduce_fn: A function that will be called on the result of the
            convolution of the PSF and ``data``.
        noise_fn: A function taking a ``PRNGKey`` and an ``Array`` and
            returning an ``Array`` of the same shape with noise applied.
    """

    optical_system: Sequence[Callable]
    reduce_fn: Optional[Callable] = None
    noise_fn: Optional[Callable[[PRNGKey, Array], Array]] = None

    def setup(self):
        """Creates the ``OpticalSystem`` describing the PSF."""
        self.psf_model = OpticalSystem(self.optical_system)

    def __call__(self, data: Array, *args: Any, **kwargs: Any) -> Array:
        """
        Computes PSF and convolves PSF with ``data`` to simulate imaging.

        Args:
            data: The sample to be imaged of shape `[B H W C]`.
            *args: Any positional arguments needed for the PSF model.
            **kwargs: Any keyword arguments needed for the PSF model.
        """
        psf = self.psf(*args, **kwargs)
        return self.image(psf, data)

    def psf(self, *args: Any, **kwargs: Any) -> Array:
        """Computes PSF intensity, taking any necessary arguments."""
        return self.psf_model(*args, **kwargs).intensity

    def output_field(self, *args: Any, **kwargs: Any) -> Field:
        """Computes PSF (complex field), taking any necessary arguments."""
        return self.psf_model(*args, **kwargs)

    def image(self, psf: Array, data: Array) -> Array:
        """
        Computes image by convolving ``psf`` and ``data``.

        If ``self.reduce_fn`` and/or ``self.noise_fn`` are defined, then they
        are called here as well.

        Args:
            psf: The PSF intensity volume to image with, has shape `[B H W C]`.
            data: The sample volume to image with, has shape `[B H W C]`.
        """
        image = vmap(fourier_convolution, in_axes=(0, 0))(data, psf)
        if self.reduce_fn is not None:
            image = self.reduce_fn(image)
        if self.noise_fn is not None:
            image = self.noise_fn(self.make_rng("noise"), image)
        return image


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
