import time

import jax.numpy as jnp
import jax.random as random
import numpy as np
from scipy.signal import czt as czt_scipy

from chromatix.utils import czt


def test_czt_1d():
    # compare by doing same computation as DFT
    key = random.PRNGKey(42)
    N = M = 100
    x = random.uniform(key, shape=(N, N))
    w = jnp.exp(-1j * 2 * jnp.pi / N)
    a = 1.0
    for axis in range(2):
        dft_x = jnp.fft.fft(x, axis=axis)
        czt_x = czt.czt_1d(x, a=a, w=w, m=M, axis=axis)
        assert jnp.allclose(dft_x, czt_x)


PLOT = True
seed = 0
np.random.seed(seed)

### scipy FFT vs. CZT
N = M = 100
axis = 0
x = np.random.randn(N, N) + 1j * np.random.randn(N, N)
x = x.astype(np.complex64)
W = np.exp(-1j * 2 * np.pi / N)
A = 1

print("\n----- scipy FFT vs scipy CZT")
dft_x = np.fft.fft(x, axis=axis)
start_time = time.time()
czt_scipy_x = czt_scipy(x, a=A, w=W, m=M, axis=axis)
assert np.allclose(dft_x, czt_scipy_x)

print("\n----- numpy FFT vs jax FFT")
dft_jax = jnp.fft.fft(x, axis=axis)
try:
    np.testing.assert_allclose(dft_jax, dft_x)
except AssertionError as e:
    print(e)

print("\n----- jax FFT vs custom CZT")
czt_x = czt.czt(x, a=A, w=W, m=M, axis=axis)
# assert np.allclose(dft_jax, czt_x)
try:
    np.testing.assert_allclose(dft_jax, czt_x)
except AssertionError as e:
    print(e)

print("\nInput dtype : ", x.dtype)
print("DFT dtype : ", dft_jax.dtype)
print("scipy CZT dtype : ", czt_scipy_x.dtype)
print("custom CZT dtype : ", czt_x.dtype)

# -- plot
if PLOT:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(dft_jax.real[axis], label="re DFT")
    plt.plot(czt_x.real[axis], label="re CZT")
    plt.plot(dft_jax.imag[axis], label="im DFT")
    plt.plot(czt_x.imag[axis], label="im CZT")
    plt.legend()
    plt.show()

# print("\n----- scipy CZT vs custom CZT")
# N = 100
# M = 10
# x = np.random.randn(N) + 1j * np.random.randn(N)
# czt_x = czt.czt(x, m=M, a=A, w=W, axis=axis)
# czt_scipy_x = czt_scipy(x, m=M, a=A, w=W, axis=axis)
# # assert jnp.allclose(czt_x, czt_scipy_x)
# try:
#     np.testing.assert_allclose(czt_scipy_x, czt_x)
# except AssertionError as e:
#     print(e)

# print("\n----- jax 2D FFT vs custom 2D CZT")
# N = 10
# M = 10
# x = np.random.randn(N, N) + 1j * np.random.randn(N, N)
# x = x.astype(np.complex64)
# W = np.exp(-1j * 2 * np.pi / N)
# A = 1
# dft_jax = jnp.fft.fft2(x)
# czt_x = czt.cztn(x, a=[A, A], w=[W, W], m=[M, M], axes=[0, 1])
# try:
#     np.testing.assert_allclose(dft_jax, czt_x)
# except AssertionError as e:
#     print(e)
