import numpy as np
import flax.linen as nn
from chex import Array
from scipy.ndimage import distance_transform_edt  # type: ignore


def sigmoid_taper(shape: tuple[int, int], width: float) -> Array:
    dist = distance_transform_edt(np.pad(np.ones((shape[0] - 2, shape[1] - 2)), 1))
    taper = 2 * (nn.sigmoid(dist / width) - 0.5)
    return taper[None, ..., None]
