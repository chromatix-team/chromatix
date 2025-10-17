import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Complex

from chromatix.typing import m


def czt(x: ArrayLike, m: int, a: Complex, w: Complex, axis: int = -1) -> Array:
    """
    Chirp Z-transform (CZT) of a signal along one dimension. The CZT is a
    generalization of the discrete Fourier transform (DFT). The DFT samples the
    Z plane at uniformly-spaced points on the unit circle, whereas the CZT
    samples the Z plane at uniformly-spaced points on a spiral. This can be
    used to interpolate the DFT to any desired frequency resolution.

    Bluestein's algorithm is used to compute the CZT as a convolution.

    !!! warning
        Using float32/complex64 may have numerical accuracy issues.

    Args:
        x: Input signal to transform.
        m: Number of samples in the output.
        a: The starting point in the complex plane. Must lie on the unit circle
            for numerical stability.
        w: The ratio between points in each step. Should lie on the unit
            circle.
        axis: Axis along which to perform the CZT.
    """

    # TODO switch to jaxtyping
    # # check input values
    # checkify.check(m > 0, "m needs to positive")
    # axis = axis + x.ndim if axis < 0 else axis
    # checkify.check(
    #     axis < x.ndim, "axis needs to be less than the number of dimensions of x"
    # )

    # compute modulation terms
    n = x.shape[axis]
    n_czt = m + n - 1
    k = jnp.arange(n_czt)
    wk2 = w ** (k**2 / 2)
    Awk2 = a ** -k[:n] * wk2[:n]
    Fwk2 = jnp.fft.fft(1 / jnp.hstack((wk2[n - 1 : 0 : -1], wk2[:m])), n_czt)
    wk2 = wk2[:m]

    # perform CZT
    x = jnp.moveaxis(x, axis, -1)
    y = jnp.fft.ifft(jnp.fft.fft(x * Awk2, n_czt, axis=-1) * Fwk2, axis=-1)
    y = y[..., n - 1 : n + m - 1] * wk2
    y = jnp.moveaxis(y, -1, axis)
    return y


def cztn(
    x: ArrayLike,
    m: tuple[int],
    a: Complex[Array, "m"],
    w: Complex[Array, "m"],
    axes: tuple[int] = (-2, -1),
) -> Array:
    """
    Chirp Z-transform (CZT) of a signal along multiple dimensions as defined by
    the `axes` parameter. This implementation loops over the dimensions and
    performs the CZT along each dimension.

    !!! warning
        Using float32/complex64 may have numerical accuracy issues.

    Args:
        x: Input signal to transform.
        m: Number of samples in the output. List for each dimension.
        a: The starting point in the complex plane. List for each dimension.
        w: The ratio between points in each step. List for each dimension.
        axes: Axes along which to perform the CZT.
    """
    x_czt = x
    for d, ax in enumerate(axes):
        x_czt = czt(x_czt, a=a[d], w=w[d], m=m[d], axis=ax)
    return x_czt


def zoomed_fft(
    x: ArrayLike,
    k_start: float,
    k_end: float,
    output_shape: tuple[int],
    axes: tuple[int] = (-2, -1),
    include_end: bool = True,
) -> ArrayLike:
    """
    Custom FFTN function that uses the Chirp Z-transform (CZT) to compute the
    Fourier transform of a signal. It allows to generate zoomed FFT in a region
    between k_start and k_end with arbitrary output shape. The usual FFT
    corresponds to output_shape = x.shape, k_start = 0, k_end = 2 * pi, and
    include_end = False.

    Args:
        x: Input signal to transform.
        k_start: Start of the frequency range.
        k_end: End of the frequency range.
        output_shape: Desired shape of the output.
        axes: Axes along which to perform the CZT.
        include_end: Whether to include the end point in the frequency range.

    Returns:
        The Fourier transform of the input signal.
    """
    if include_end:
        renorm = tuple(m - 1 for m in output_shape)
    else:
        renorm = output_shape
    w = tuple(jnp.exp(1j * (k_end - k_start) / n) for n in renorm)

    a = jnp.exp(-1j * k_start)
    a = tuple(a for _ in range(len(output_shape)))

    return cztn(
        x=x,
        m=output_shape,
        a=a,
        w=w,
        axes=axes,
    )
