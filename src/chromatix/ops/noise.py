import jax.numpy as jnp
from jax import random, custom_jvp


@custom_jvp
def approximate_shot_noise(key, image):
    """Approximates shot noise with a Gaussian with std=sqrt(N)"""
    noisy = image + jnp.sqrt(image) * random.normal(key, image.shape)
    return jnp.maximum(noisy, 0.0)


@approximate_shot_noise.defjvp
def approximate_shotnoise_jvp(primals, tangents):
    """Custom backprop for approximate shotnoise"""
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
def shot_noise(key, image):
    """Shot noise with custom backpro approximating the gradient as gradient
    of gaussian with std=sqrt(N)."""
    noisy = random.poisson(key, image, image.shape)
    return jnp.float32(noisy)


@shot_noise.defjvp
def shotnoise_jvp(primals, tangents):
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
