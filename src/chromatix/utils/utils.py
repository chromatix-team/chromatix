import jax.numpy as jnp
import numpy as np
from chex import Array
from typing import Sequence
import flax.linen as nn
from scipy.ndimage import distance_transform_edt  # type: ignore
from typing import Tuple
from .shapes import _broadcast_2d_to_spatial


def next_order(val: int) -> int:
    return int(2 ** np.ceil(np.log2(val)))


def center_pad(u: jnp.ndarray, pad_width: Sequence[int], cval: float = 0) -> Array:
    """
    Symmetrically pads ``u`` with lengths specified per axis in ``n_padding``,
    which should be iterable and have the same size as ``u.ndims``.
    """
    pad = [(n, n) for n in pad_width]
    return jnp.pad(u, pad, constant_values=cval)


def center_crop(u: jnp.ndarray, crop_length: Sequence[int]) -> Array:
    """
    Symmetrically crops ``u`` with lengths specified per axis in
    ``crop_length``, which should be iterable with same size as ``u.ndims``.
    """
    crop_length = [0 if length is None else length for length in crop_length]
    crop = tuple([slice(n, size - n) for size, n in zip(u.shape, crop_length)])
    return u[crop]


def sigmoid_taper(shape: Tuple[int, int], width: float, ndim: int = 5) -> Array:
    dist = distance_transform_edt(np.pad(np.ones((shape[0] - 2, shape[1] - 2)), 1))
    taper = 2 * (nn.sigmoid(dist / width) - 0.5)
    return _broadcast_2d_to_spatial(taper, ndim)
