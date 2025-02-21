import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import jit
from scipy.signal import czt as czt_scipy

from chromatix.utils.czt import czt, cztn

# tests fail for float32/complex64
jax.config.update("jax_enable_x64", True)
key = random.PRNGKey(42)


def test_czt_as_dft():
    # compare by doing same computation as DFT
    N = M = 100
    x = random.uniform(key, shape=(N, N)) + 1j * random.uniform(key, shape=(N, N))
    w = jnp.exp(-1j * 2 * jnp.pi / N)
    for axis in range(2):
        dft_x = jnp.fft.fft(x, axis=axis)
        czt_x = czt(x, a=1.0, w=w, m=M, axis=axis)
        assert jnp.allclose(dft_x, czt_x)


def test_czt_as_idft():
    # compare by doing same computation as DFT
    N = M = 100
    x = random.uniform(key, shape=(N, N)) + 1j * random.uniform(key, shape=(N, N))
    w = jnp.exp(1j * 2 * jnp.pi / N)
    for axis in range(2):
        dft_x = jnp.fft.ifft(x, axis=axis)
        czt_x = czt(x, a=1.0, w=w, m=M, axis=axis)
        assert jnp.allclose(dft_x, czt_x / N)  # czt does not do the scaling.


def test_cztn_as_dftn():
    N = M = 10
    x = random.uniform(key, shape=(N, N, N)) + 1j * random.uniform(key, shape=(N, N, N))
    w = jnp.exp(-1j * 2 * jnp.pi / N)
    dft_x = jnp.fft.fftn(x, axes=(1, 2))
    czt_x = cztn(x, a=(1, 1), w=(w, w), m=(M, M), axes=(1, 2))
    assert jnp.allclose(dft_x, czt_x)


def test_against_scipy():
    N = 100
    M = 10
    x = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    w = np.exp(-1j * 2 * np.pi / N)
    a = 1.0
    for axis in range(2):
        ref = czt_scipy(x, a=a, w=w, m=M, axis=axis)
        czt_chromatix = czt(x, a=a, w=w, m=M, axis=axis)
        assert np.allclose(ref, czt_chromatix)


# check jittable
def test_czt_jit():
    N = M = 10
    x = random.uniform(key, shape=(N, N)) + 1j * random.uniform(key, shape=(N, N))
    w = jnp.exp(-1j * 2 * jnp.pi / N)
    # -- czt
    czt_jit = jit(czt, static_argnums=[1])
    czt_jit(x, a=1.0, w=w, m=M)
    czt_jit = jit(czt, static_argnums=[1, 4])
    czt_jit(x, a=1.0, w=w, m=M, axis=0)
    # -- cztn
    cztn_jit = jit(cztn, static_argnums=[1])
    cztn_jit(x, a=(1, 1), w=(w, w), m=(M, M))
    cztn_jit = jit(cztn, static_argnums=[1, 4])
    cztn_jit(x, a=(1, 1), w=(w, w), m=(M, M), axes=(0, 1))
