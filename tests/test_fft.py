import jax.numpy as jnp
from jax import random
from chromatix.ops.fft import fft, ifft


def test_looped_fft():
    key = random.PRNGKey(42)
    x = random.normal(key, (1, 256, 256, 10), dtype=jnp.complex64)
    assert jnp.allclose(fft(x), fft(x, loop_axis=-1))


def test_looped_ifft():
    key = random.PRNGKey(42)
    x = random.normal(key, (1, 256, 256, 10), dtype=jnp.complex64)
    assert jnp.allclose(ifft(x), ifft(x, loop_axis=-1))
