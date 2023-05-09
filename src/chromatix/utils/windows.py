import numpy as np
import flax.linen as nn
from chex import Array
from scipy.ndimage import distance_transform_edt  # type: ignore
from typing import Tuple
from . import _broadcast_2d_to_spatial


def sigmoid_taper(shape: Tuple[int, int], width: float, ndim: int = 5) -> Array:
    dist = distance_transform_edt(np.pad(np.ones((shape[0] - 2, shape[1] - 2)), 1))
    taper = 2 * (nn.sigmoid(dist / width) - 0.5)
    return _broadcast_2d_to_spatial(taper, ndim)
