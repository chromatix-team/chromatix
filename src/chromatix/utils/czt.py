from typing import List

import jax.numpy as jnp
from chex import Array


def _verify_axes(x, axes):
    axes = [a + x.ndim if a < 0 else a for a in axes]
    if any(a >= x.ndim or a < 0 for a in axes):
        raise ValueError("axes exceeds dimensionality of input")
    if len(set(axes)) != len(axes):
        raise ValueError("all axes must be unique")
    return axes


def _verify_cztn_input(x, a, w, m, axes):
    D = len(a)
    if not (len(w) == len(m) == D):
        raise ValueError("Length of [a], [w], and [m] must match.")

    if x.ndim < D:
        raise ValueError("[x] does not have enough dimensions.")

    if axes is not None:
        assert len(axes) == D, "Length of [axes] must match [a], [w], and [m]."
        axes = _verify_axes(x, axes)
    else:
        axes = list(range(D))

    # check valid values
    for d in range(D):
        a[d] = complex(a[d])
        w[d] = complex(w[d])
        if not jnp.isclose(abs(a[d]), 1):
            raise ValueError(
                "Parameter[a[d]] must lie on the unit circle for numerical stability."
            )
        if not jnp.isclose(abs(w[d]), 1):
            raise ValueError("Parameter[w[d]] must lie on the unit circle.")
        if m[d] <= 0:
            raise ValueError("Parameter[m[d]] must be positive.")

    return axes, a, w


def czt(x: Array, m: int, a: complex, w: complex, axis=-1) -> Array:
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
        a: The starting point in the complex plane.
        w: The ratio between points in each step.
    """
    axes, a, w = _verify_cztn_input(x, a=[a], w=[w], m=[m], axes=[axis])
    axis = axes[0]
    a = a[0]
    w = w[0]

    # compute modulation terms
    n = x.shape[axis]
    n_czt = m + n - 1
    k = jnp.arange(n_czt)
    wk2 = w ** (k**2 / 2)
    Awk2 = a ** -k[:n] * wk2[:n]
    Fwk2 = jnp.fft.fft(1 / jnp.hstack((wk2[n - 1 : 0 : -1], wk2[:m])), n_czt)
    wk2 = wk2[:m]
    idx = [slice(None)] * len(x.shape)
    idx[axis] = slice(n - 1, n + m - 1)
    idx = tuple(idx)

    # perform CZT
    broadcast_shape = [1] * len(x.shape)
    broadcast_shape[axis] = n
    y = jnp.fft.fft(x * Awk2.reshape(broadcast_shape), n_czt, axis=axis)
    broadcast_shape[axis] = n_czt
    y = jnp.fft.ifft(y * Fwk2.reshape(broadcast_shape), axis=axis)
    broadcast_shape[axis] = m
    y = y[idx] * wk2.reshape(broadcast_shape)
    return y


def cztn(x: Array, m: List, a: List, w: List, axes: List) -> Array:
    """
    Chirp Z-transform (CZT) of a signal along multiple dimensions as defined by
    the `axes` parameter.

    !!! warning
        Using float32/complex64 may have numerical accuracy issues.

    Args:
        x: Input signal to transform.
        m: Number of samples in the output. List for each dimension.
        a: The starting point in the complex plane. List for each dimension.
        w: The ratio between points in each step. List for each dimension.
        axes: Axes along which to perform the CZT.
    """

    axes, a, w = _verify_cztn_input(x, a=a, w=w, m=m, axes=axes)

    # initialize variables and modulation vectors
    n_dim = len(axes)
    all_dims = tuple([d for d in range(len(x.shape))])
    n_czt = []
    Awk2 = []
    Fwk2 = []
    wk2 = []
    for d in range(n_dim):
        N = x.shape[d]
        M = m[d]
        n_czt.append(N + M - 1)

        # pre-compute modulation vectors
        k = jnp.arange(n_czt[d])
        _wk2 = w[d] ** (k**2 / 2)
        _wk2_neg = w[d] ** (-(k**2) / 2)
        Awk2.append((a[d] ** -k[:N]) * _wk2[:N])
        Fwk2.append(
            jnp.fft.fft(
                jnp.concatenate(
                    [
                        _wk2_neg[:M],
                        jnp.zeros(n_czt[d] - M - N + 1),
                        _wk2_neg[(N - 1) : 0 : -1],
                    ]
                ),
            )
        )
        wk2.append(_wk2[:M])

    # modulate input (chirp)
    mod_args = []
    for d in range(n_dim):
        mod_args.append(Awk2[d])
        mod_args.append((..., axes[d]))
    _x = jnp.einsum(x, all_dims, *mod_args, all_dims)
    _x = jnp.fft.fftn(_x, axes=axes, s=n_czt)

    # convolution
    mod_args = []
    for d in range(n_dim):
        mod_args.append(Fwk2[d])
        mod_args.append((..., axes[d]))
    _x = jnp.einsum(_x, all_dims, *mod_args, all_dims)
    _x = jnp.fft.ifftn(_x, axes=axes)

    # crop desired region
    slices = [slice(None)] * len(_x.shape)
    for d in range(n_dim):
        slices[axes[d]] = slice(m[d])
    _x = _x[tuple(slices)]

    # final modulation terms
    mod_args = []
    for d in range(n_dim):
        mod_args.append(wk2[d])
        mod_args.append((..., axes[d]))
    y = jnp.einsum(_x, all_dims, *mod_args, all_dims)
    return y


def _czt(x: Array, m: int, a: complex, w: complex, axis=-1) -> Array:
    """
    1D CZT by calling multi-dimensional one (slower).
    """
    return cztn(x, m=[m], a=[a], w=[w], axes=[axis])


def _cztn(x: Array, m: List, a: List, w: List, axes: List) -> Array:
    """
    Slower version of multi-dimension CZT as a sequence of 1D CZT.
    """
    x_czt = x.copy()
    for d, ax in enumerate(axes):
        x_czt = czt(x_czt, a=a[d], w=w[d], m=m[d], axis=ax)
    return x_czt
