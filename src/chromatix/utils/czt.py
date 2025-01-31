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
        raise ValueError("Length of [A], [W], and [M] must match.")

    if x.ndim < D:
        raise ValueError("[x] does not have enough dimensions.")

    if axes is not None:
        assert len(axes) == D, "Length of [axes] must match [A], [W], and [M]."
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
    Chirp Z-transform (CZT) of a signal along one dimension.

    Args:
        x: Input signal to transform.
        m: Number of samples in the output.
        a: The starting point in the complex plane.
        w: The ratio between points in each step.
    """
    return cztn(x, m=[m], a=[a], w=[w], axes=[axis])


def cztn(x: Array, m: List, a: List, w: List, axes: List) -> Array:
    """
    Chirp Z-transform (CZT) of a signal along multiple dimensions.

    Args:
        x: Input signal to transform.
        m: Number of samples in the output. List for each dimension.
        a: The starting point in the complex plane. List for each dimension.
        w: The ratio between points in each step. List for each dimension.
        axes: Axes along which to perform the CZT.
    """

    axes, a, w = _verify_cztn_input(x, a=a, w=w, m=m, axes=axes)

    # initialize variables
    n_dim = len(axes)
    n_input = [x.shape[d] for d in axes]
    n_czt = []
    n_idx = []
    for d in range(n_dim):
        n_czt.append(n_input[d] + m[d] - 1)  # TODO next fast len or not?
        # from scipy.fft import next_fast_len
        # n_czt.append(next_fast_len(n_input[d] + m[d] - 1))
        n_idx.append(jnp.arange(n_czt[d]))

    # modulate input (chirp)
    u = x.copy()  # TODO copy?
    for d in range(n_dim):
        broadcast_shape = [1] * len(x.shape)
        broadcast_shape[axes[d]] = n_input[d]
        u_mod_d = (a[d] ** -n_idx[d][: n_input[d]]) * (
            w[d] ** (n_idx[d][: n_input[d]] ** 2 / 2)
        )
        u *= u_mod_d.reshape(broadcast_shape)
    U = jnp.fft.fftn(u, axes=axes, s=n_czt)

    # prepare second sequence for convolution
    for d in range(n_dim):
        _n = n_input[d]
        broadcast_shape = [1] * len(x.shape)
        broadcast_shape[axes[d]] = n_czt[d]
        v_left = w[d] ** (-(n_idx[d][: m[d]] ** 2) / 2)
        v_center = jnp.zeros(n_czt[d] - m[d] - _n + 1)
        v_right = w[d] ** (-((n_czt[d] - n_idx[d][n_czt[d] - _n + 1 :]) ** 2) / 2)
        v = jnp.concatenate((v_left, v_center, v_right))
        V = jnp.fft.fft(v).reshape(broadcast_shape)
        U *= V
    out = jnp.fft.ifftn(U, axes=axes)

    # crop desired region
    slices = [slice(None)] * len(out.shape)
    for d in range(n_dim):
        slices[axes[d]] = slice(m[d])
    out = out[tuple(slices)]

    # final modulation terms
    for d in range(n_dim):
        broadcast_shape = [1] * len(x.shape)
        broadcast_shape[axes[d]] = m[d]
        czt_mod = w[d] ** (n_idx[d][: m[d]] ** 2 / 2)
        out *= czt_mod.reshape(broadcast_shape)

    return out
