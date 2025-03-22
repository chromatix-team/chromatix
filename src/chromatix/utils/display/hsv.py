import numpy as np
from typing import Union, Sequence


def hsv2rgb(hue: Union[float, Sequence, np.ndarray] = None,
            saturation: Union[float, Sequence, np.ndarray] = None,
            value: Union[float, Sequence, np.ndarray] = None,
            alpha: Union[float, Sequence, np.ndarray] = None,
            axis: int = -1) -> np.ndarray:
    """
    Converts a hue-saturation-intensity value image to a red-green-blue image.

    :param hue: A 2D numpy.array with the hue.
        If the saturation is not provided, the first argument will be interpreted as a 3D numpy.array with the HSV image,
        the channel must be in the final right-hand dimension.
    :param saturation: (optional) a 2D numpy.array with the saturation.
    :param value: (optional) a 2D numpy.array with the intensity value.
    :param alpha: (optional) a 2D numpy.array with the opaqueness alpha value.
    :param axis: (default: -1) The channel axis of the output array, and also the input array if neither saturation, nor
        value are provided.
    :return: rgb_image: a 3D numpy.array with the RGB image, the channel in the final right-hand dimension, or axis if provided.
    """
    if saturation is None and value is None:
        hsv_image = np.moveaxis(hue, axis, 0)
        if alpha is None and hsv_image.shape[0] > 3:
            alpha = hsv_image[3]
        value = hsv_image[2]
        saturation = hsv_image[1]
        hue = hsv_image[0]
    else:
        hue = np.asarray(hue)
        saturation = np.asarray(saturation)
        value = np.asarray(value)

    hue = 6.0 * hue
    i = hue.astype(np.int8)  # integer level
    f = hue - i  # fractional level
    i %= 6

    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - (saturation * (1.0 - f)))

    r = ((i == 0) | (i == 5)) * value + (i == 1) * q + ((i == 2) | (i == 3)) * p + (i == 4) * t
    g = (i == 0) * t + ((i == 1) | (i == 2)) * value + (i == 3) * q + (i >= 4) * p
    b = (i <= 1) * p + (i == 2) * t + ((i == 3) | (i == 4)) * value + (i == 5) * q
    if alpha is None:
        rgb_image = np.stack(np.broadcast_arrays(r, g, b))
    else:
        rgb_image = np.stack(np.broadcast_arrays(r, g, b, alpha))

    return np.moveaxis(rgb_image, 0, axis)


def rgb2hsv(red: Union[float, Sequence, np.ndarray],
            green: Union[float, Sequence, np.ndarray] = None,
            blue: Union[float, Sequence, np.ndarray] = None,
            alpha: Union[float, Sequence, np.ndarray] = None,
            axis=-1) -> np.ndarray:
    """
    Converts a red-green-blue value image to a hue-saturation-intensity image.

    :param red: An 2D numpy.array with the red channel.
        If neither green and blue are provided, then this will be interpreted as a stack of the red, green, and blue channels.
    :param green: (optional) a 2D numpy.array with the green channel.
    :param blue: (optional) a 2D numpy.array with the blue channel.
    :param alpha: (optional) a 2D numpy.array with the opaqueness alpha value.
    :param axis: (default: -1) The channel axis of the output array, and also the input array if neither saturation, nor
        value are provided.

    :return: hsv_image: a 3D numpy.array with the HSV image
    """
    if green is None and blue is None:
        rgb_image = np.moveaxis(red, axis, 0)
    elif alpha is None:
        rgb_image = np.stack(np.broadcast_arrays(red, green, blue))
    else:
        rgb_image = np.stack(np.broadcast_arrays(red, green, blue, alpha))

    v = np.amax(rgb_image, axis=0)
    s_x_v = v - np.amin(rgb_image, axis=0)

    def per_pixel_select(arr, idx):
        selection = np.zeros(arr.shape[1:], dtype=arr.dtype)
        for channel in range(rgb_image.shape[0]):
            selection += arr[channel] * (idx == channel)
        return selection

    ch1 = (rgb_image[0] < rgb_image[1]) * (rgb_image[0] <= rgb_image[2])
    ch2 = (rgb_image[1] < rgb_image[2]) * (rgb_image[1] <= rgb_image[0])
    channel = np.array(ch1 + 2 * ch2)   # dominant channel
    dominant = per_pixel_select(rgb_image, channel)  # mix from channel,
    after = per_pixel_select(rgb_image, np.mod(channel + 1, 3))  # to channel,
    before = per_pixel_select(rgb_image, np.mod(channel - 1, 3))  # and the other channel
    i = 2 * channel + (dominant <= after) * (1 - np.isclose(after, before))  # interval index in range(6)

    f = np.mod((dominant - after) / (s_x_v + np.isclose(s_x_v, 0)), 1)  # fraction in interval
    # 1 0 0   0/6  2
    # 1 1 0   1/6  2
    # 0 1 0   2/6  0
    # 0 1 1   3/6  0
    # 0 0 1   4/6  1
    # 1 0 1   5/6  1
    # 1 0 0   6/6
    h = (i + f) / 6

    s = s_x_v / (v + np.isclose(v, 0))

    if alpha is None:
        hsv_image = np.stack((h, s, v))
    else:
        hsv_image = np.stack((h, s, v, alpha))

    return np.moveaxis(hsv_image, 0, axis)
