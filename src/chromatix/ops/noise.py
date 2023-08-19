import jax.numpy as jnp
from jax import random, custom_jvp
from chex import Array, PRNGKey
from typing import Tuple


@custom_jvp
def approximate_shot_noise(key: PRNGKey, image: Array) -> Array:
    """
    Approximates Poisson shot noise using a Gaussian for differentiability.
    """
    noisy = image + jnp.sqrt(image) * random.normal(key, image.shape)
    return jnp.maximum(noisy, 0.0)


@approximate_shot_noise.defjvp
def approximate_shotnoise_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    """
    Custom gradient for ``approximate_shot_noise``.

    This is necessary to fix an instability when the input ``image`` is 0.
    """
    key, image = primals
    _, image_dot = tangents
    primal_out = approximate_shot_noise(key, image)
    # We define the gradient to be zero if image=0
    # we just add eta as we multiply by zero later anyway
    noise_grad = jnp.ones_like(image) + random.normal(key, image.shape) / (
        2 * jnp.sqrt(image) + 1e-6
    )
    # maximum operation, abs to get rid of -0
    tangent_out = image_dot * jnp.abs(noise_grad) * (primal_out != 0)
    return primal_out, tangent_out


@custom_jvp
def shot_noise(key: PRNGKey, image: Array) -> Array:
    """
    Simulates Poisson shot noise whose gradient is approximated using
    the gradient of a Gaussian, just as if the simulation had been
    ``approximate_shot_noise`` instead.
    """
    noisy = random.poisson(key, image, image.shape)
    return jnp.float32(noisy)


@shot_noise.defjvp
def shotnoise_jvp(primals: Tuple, tangents: Tuple) -> Tuple:
    """
    Custom gradient for ``shot_noise``.

    Because the Poisson distribution computed in ``shot_noise`` cannot be
    differentiated, this function computes the gradient as if the forward pass
    had been ``approximate_shot_noise``.
    """
    key, image = primals
    _, image_dot = tangents
    primal_out = shot_noise(key, image)
    # We define the gradient to be zero if image=0
    # we just add eta as we multiply by zero later anyway
    noise_grad = jnp.ones_like(image) + random.normal(key, image.shape) / (
        2 * jnp.sqrt(image) + 1e-6
    )
    # maximum operation, abs to get rid of -0)
    tangent_out = image_dot * jnp.abs(noise_grad) * (primal_out != 0)
    return primal_out, tangent_out
