from typing import Callable, Optional, Tuple, Union

import flax.linen as nn
from chex import PRNGKey

from chromatix.elements.utils import register

from .. import functional as cf
from ..field import Field

__all__ = ["ThinLens", "FFLens", "DFLens", "HighNALens"]


class ThinLens(nn.Module):
    """
    Applies a thin lens placed directly after the incoming ``Field``.
    This element returns the ``Field`` directly after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    The attributes ``f``, ``n``, and ``NA`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
    """

    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Optional[Union[float, Callable[[PRNGKey], float]]] = None

    @nn.compact
    def __call__(self, field: Field) -> Field:
        f = register(self, "f")
        n = register(self, "n")
        NA = register(self, "NA")
        return cf.thin_lens(field, f, n, NA)


class FFLens(nn.Module):
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.
    This element returns the ``Field`` a distance ``f`` after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    The attributes ``f``, ``n``, and ``NA`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether to use IFFT (default is False, which uses FFT).
    """

    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Optional[Union[float, Callable[[PRNGKey], float]]] = None
    inverse: bool = False

    @nn.compact
    def __call__(self, field: Field) -> Field:
        f = register(self, "f")
        n = register(self, "n")
        NA = register(self, "NA")
        return cf.ff_lens(field, f, n, NA, inverse=self.inverse)


class HighNALens(nn.Module):
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.
    This element returns the ``Field`` a distance ``f`` after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    The attributes ``f``, ``n``, and ``NA`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        camera_shape: The shape of the camera (in pixels).
        camera_pixel_pitch: The pixel pitch of the camera (in microns).
        wavelength: The wavelength of the light (in microns).
    """

    NA: Optional[Union[float, Callable[[PRNGKey], float]]]
    camera_shape: Tuple[int, int]
    camera_pixel_pitch: Union[float, Callable[[PRNGKey], float]]
    wavelength: Union[float, Callable[[PRNGKey], float]]

    @nn.compact
    def __call__(self, field: Field) -> Field:
        NA = register(self, "NA")
        camera_shape = register(self, "camera_shape")
        camera_pixel_pitch = register(self, "camera_pixel_pitch")
        wavelength = register(self, "wavelength")
        return cf.lenses.high_na_lens(
            field=field,
            NA=NA,
            output_shape=camera_shape,
            output_dx=camera_pixel_pitch,
            wavelength=wavelength,
        )


class DFLens(nn.Module):
    """
    Applies a thin lens placed a distance ``d`` after the incoming ``Field``.
    This element returns the ``Field`` a distance ``f`` after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    The attributes ``d``, ``f``, ``n``, and ``NA`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        d: Distance from the incoming ``Field`` to the lens.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether to use IFFT (default is False, which uses FFT).
    """

    d: Union[float, Callable[[PRNGKey], float]]
    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Optional[Union[float, Callable[[PRNGKey], float]]] = None
    inverse: bool = False

    @nn.compact
    def __call__(self, field: Field) -> Field:
        d = register(self, "d")
        f = register(self, "f")
        n = register(self, "n")
        NA = register(self, "NA")
        return cf.df_lens(field, d, f, n, NA, inverse=self.inverse)
