import jax.numpy as jnp
from jax.nn.initializers import lecun_normal
from jax.lax import complex
from jax.random import KeyArray
import numpy as np

from typing import Any, Sequence, Callable, Optional


def trainable(x: Any) -> Callable:
    """
    Returns a function with a valid signature for a Flax parameter initializer
    function (accepts a jax.random.KeyArray), which simply returns x itself. When a
    Chromatix element is constructed with such a function as its attribute, it will
    automatically turn that into a parameter to be optimized. Thus, this function
    is a convenient way to set the attribute of an optical element in Chromatix as
    a trainable parameter initialized to value x.
    """

    def init_fn(key: KeyArray, *args, **kwargs) -> Any:
        return x

    return init_fn


def next_order(val: int) -> int:
    return int(2 ** np.ceil(np.log2(val)))


def complex_lecun_normal(*args, **kwargs) -> Callable:
    """Thin wrapper so lecun normal returns complex number."""

    def _init(key, shape: Sequence[int]) -> jnp.ndarray:
        return complex(*normal(key, (2, *shape)))

    normal = lecun_normal(*args, **kwargs)
    return _init


def center_pad(u: jnp.ndarray, pad_width: Sequence[int]) -> jnp.ndarray:
    """Symmetrically pads u with lengths specified per axis in n_padding.
    n_padding should be iterable with same size as u.ndims."""
    pad = [(n, n) for n in pad_width]
    return jnp.pad(u, pad)


def center_crop(u: jnp.ndarray, crop_length: Sequence[int]) -> jnp.ndarray:
    """
    Symmetrically crops u with lengths specified per axis in
    crop_length, which should be iterable with same size as u.ndims.
    """
    crop_length = [0 if length is None else length for length in crop_length]
    crop = tuple([slice(n, size - n) for size, n in zip(u.shape, crop_length)])
    return u[crop]


def gaussian_kernel(
    sigma: Sequence[float], truncate: float = 4.0, shape: Optional[Sequence[int]] = None
) -> jnp.ndarray:
    """Returns N-D gaussian kernel"""
    _sigma = np.atleast_1d(np.array(sigma))
    if shape is not None:
        _shape = np.atleast_1d(np.array(shape))
        assert np.all(_shape % 2 != 0), "Shape should be uneven"
        radius = ((_shape - 1) / 2).astype(np.int16)
    else:
        radius = (truncate * _sigma + 0.5).astype(np.int16)

    x = jnp.mgrid[tuple(slice(-r, r + 1) for r in radius)]
    phi = jnp.exp(-0.5 * jnp.sum((x.T / _sigma) ** 2, axis=-1))  # type: ignore
    return phi / phi.sum()
