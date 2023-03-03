from typing import Tuple
import jax
import jax.numpy as jnp
from chex import Array

__all__ = ["binarize", "binarize_jvp"]


@jax.custom_jvp
def binarize(x: Array) -> Array:
    """
    Binarize each pixel of amplitude mask to be either 0 or 1
    """
    return (x > 0.5) * 1.0


@binarize.defjvp
def binarize_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    """
    Custom gradient for ``binarize``.

    We approximate the gradient of ``binarize`` function with the gradient of the sigmoid function
    """
    alpha = 1
    sig = 1 / (1 + jnp.exp((-alpha * primals[0]) + 0.5 * alpha))
    return binarize(primals[0]), tangents[0] * alpha * (sig - sig**2)
