"""
Code ported from
[this file in MacroMax](https://github.com/tttom/MacroMax/blob/master/python/macromax/utils/display/complex2rgb.py) and
adapted for JAX.
"""
import jax
import jax.numpy as jnp
from typing import Union, Sequence, Optional

from .hsv import hsv2rgb


__all__ = ['complex2rgb']


def complex2rgb(complex_image: Union[complex, Sequence, jax.Array], normalization: Union[bool, float, int] = None,
                inverted:bool = False, alpha: Optional[float] = None, dtype=float, axis: int = -1) -> jax.Array:
    """
    Converts a complex image to an RGB image.

    :param complex_image: A 2D array
    :param normalization: An optional multidimensional to indicate the target magnitude of the maximum value
        (1.0 is saturation).
    :param inverted: By default, 0 is shown as black and amplitudes of 1 as the brightest hues. Setting this input
        argument to True could be useful for printing on a white background.
    :param alpha: The maximum alpha value (0 = transparent, 1 is opaque). When specified, each pixel's alpha value is
        proportional to the intensity times this value. Default: None = no alpha channel.
    :param dtype: The output data type. The value is scaled to the maximum positive numeric range for integers
        (jnp.iinfo(dtype).max). Floating point numbers are within [0, 1]. (Default: float)
    :param axis: (default: -1) The channel axis of the output array, and also the input array if neither saturation, nor
        value are provided.

    :return: A real 3d-array with values between 0 and 1 if the dtype is a float and covering the dynamic range when
        the dtype is an integer.
    """
    # Make sure that this is a numpy 2d-array
    complex_image = jnp.asarray(complex_image)

    amplitude = jnp.abs(complex_image)
    if amplitude.dtype == bool:
        amplitude = amplitude.astype(float)
    phase = jnp.angle(complex_image)

    if normalization is not None:
        if normalization > 0:
            max_value = jnp.amax(jnp.abs(amplitude))
            if max_value > 0:
                amplitude *= normalization / max_value

    hue = phase / (2 * jnp.pi) + 0.5
    clipped_amplitude = jnp.minimum(amplitude, 1.0)
    if not inverted:
        intensity = clipped_amplitude
        saturation = jnp.clip(2.0 - amplitude, 0.0, 1.0)
    else:
        intensity = jnp.maximum(0.0, 1.0 - 0.5 * amplitude)
        saturation = jnp.clip(amplitude, 0.0, 1.0)

    if alpha is not None:
        alpha = intensity * alpha

    rgb_image = hsv2rgb(hue, saturation, intensity, alpha, axis=axis)  # Convert HSV to an RGB image (optionally with alpha channel)

    if issubclass(dtype, jnp.integer):
        rgb_image = rgb_image * jnp.iinfo(dtype).max + 0.5

    return rgb_image.astype(dtype)
