"""Banded separable resampling: a sparse alternative to a dense weight matrix.

``jax.image.scale_and_translate`` builds a dense ``(in_size, out_size)`` weight
matrix per axis and contracts it, i.e. O(in * out) work per axis, even though
every interpolation kernel here has only a few nonzero taps per output:

    linear    radius 1   ->  ~3 taps
    cubic     radius 2   ->  ~5 taps
    lanczos3  radius 3   ->  ~7 taps
    lanczos5  radius 5   ->  ~11 taps

(multiplied by ``ceil(1/scale)`` when antialiasing a downsample, since the
kernel is stretched to low-pass first).

This module gathers only that band, and gives the gather a ``custom_vjp`` whose
backward is *another* banded gather using the exact transpose band, rather than
autodiff's scatter-add over data-dependent indices.

Weight construction deliberately mirrors ``jax._src.image.scale.compute_weight_mat``
(same kernels, same two normalisation guards, same out-of-range masking) so the
two paths agree to float32 accumulation noise.

The forward and transpose bands need **different** static widths: the forward
band spans ``2*radius*kernel_scale`` inputs (wide when downsampling) while the
transpose spans ``2*radius*max(1, scale)`` outputs (wide when upsampling).
Sizing both from the forward width silently truncates the adjoint when
upsampling -- values stay correct while gradients are quietly wrong.
"""

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["banded_scale_and_translate", "KERNELS"]


def _lanczos(radius: int):
    def kernel(x):
        y = radius * jnp.sin(np.pi * x) * jnp.sin(np.pi * x / radius)
        out = jnp.where(
            x > 1e-3, jnp.divide(y, jnp.where(x != 0, np.pi**2 * x**2, 1)), 1
        )
        return jnp.where(x > radius, 0.0, out)

    return kernel


def _triangle(x):
    return jnp.maximum(0, 1 - jnp.abs(x))


def _keys_cubic(x):
    # Keys kernel with A = -0.5, matching jax.image's "cubic".
    out = ((1.5 * x - 2.5) * x) * x + 1.0
    out = jnp.where(x >= 1.0, ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0, out)
    return jnp.where(x >= 2.0, 0.0, out)


#: Interpolation method -> (kernel function, radius).
KERNELS = {
    "linear": (_triangle, 1),
    "cubic": (_keys_cubic, 2),
    "lanczos3": (_lanczos(3), 3),
    "lanczos5": (_lanczos(5), 5),
}

_EPS = 1000.0 * float(np.finfo(np.float32).eps)


def _forward_width(radius: int, in_size: int, out_size: int, slack: int = 2) -> int:
    """Static tap count covering the forward band (inputs read per output)."""
    return 2 * int(np.ceil(radius * max(in_size / out_size, 1.0))) + 1 + slack


def _transpose_width(radius: int, in_size: int, out_size: int, slack: int = 2) -> int:
    """Static tap count covering the transpose band (outputs fed per input)."""
    return 2 * int(np.ceil(radius * max(out_size / in_size, 1.0))) + 1 + slack


def _forward_band(in_size, out_size, scale, translation, kernel, radius, width):
    """Per output ``o``: clamped input indices and weights (0 off-support)."""
    inv_scale = 1.0 / scale
    kernel_scale = jnp.maximum(inv_scale, 1.0)  # antialias=True
    o = jnp.arange(out_size, dtype=jnp.float32)
    sample_f = (o + 0.5) * inv_scale - translation * inv_scale - 0.5

    base = jnp.floor(sample_f - radius * kernel_scale).astype(jnp.int32)
    idx = base[:, None] + jnp.arange(width, dtype=jnp.int32)
    dist = jnp.abs(sample_f[:, None] - idx.astype(jnp.float32)) / kernel_scale
    w = jnp.where((idx >= 0) & (idx < in_size), kernel(dist), 0.0)

    # Normalising over the band equals jax's full-column normalisation, since
    # every tap outside the band is exactly zero.
    total = jnp.sum(w, axis=1, keepdims=True)
    w = jnp.where(
        jnp.abs(total) > _EPS, jnp.divide(w, jnp.where(total != 0, total, 1)), 0.0
    )
    valid = (sample_f >= -0.5) & (sample_f <= (in_size - 0.5))
    return jnp.clip(idx, 0, in_size - 1), jnp.where(valid[:, None], w, 0.0)


def _transpose_band(
    in_size, out_size, scale, translation, kernel, radius, width, forward_width
):
    """Per input ``j``: the outputs that read it, with the adjoint weights."""
    inv_scale = 1.0 / scale
    kernel_scale = jnp.maximum(inv_scale, 1.0)
    j = jnp.arange(in_size, dtype=jnp.float32)
    centre_o = (j + 0.5) * scale + translation - 0.5

    base = jnp.floor(centre_o - radius * kernel_scale * scale).astype(jnp.int32)
    idx_t = base[:, None] + jnp.arange(width, dtype=jnp.int32)
    sample_f_o = (
        (idx_t.astype(jnp.float32) + 0.5) * inv_scale - translation * inv_scale - 0.5
    )
    dist = jnp.abs(sample_f_o - j[:, None]) / kernel_scale
    w_raw = jnp.where((idx_t >= 0) & (idx_t < out_size), kernel(dist), 0.0)

    # The forward normalisation is per OUTPUT, so the adjoint weight for
    # (input j -> output o) must be divided by output o's own column sum --
    # recomputed here over the FORWARD width, not the transpose width.
    base_in = jnp.floor(sample_f_o - radius * kernel_scale).astype(jnp.int32)
    j_idx = base_in[..., None] + jnp.arange(forward_width, dtype=jnp.int32)
    dist_col = jnp.abs(sample_f_o[..., None] - j_idx.astype(jnp.float32)) / kernel_scale
    col_sum = jnp.sum(
        jnp.where((j_idx >= 0) & (j_idx < in_size), kernel(dist_col), 0.0), axis=-1
    )
    w_t = jnp.where(
        jnp.abs(col_sum) > _EPS,
        jnp.divide(w_raw, jnp.where(col_sum != 0, col_sum, 1)),
        0.0,
    )
    valid = (sample_f_o >= -0.5) & (sample_f_o <= (in_size - 0.5))
    return jnp.clip(idx_t, 0, out_size - 1), jnp.where(valid, w_t, 0.0)


def _gather(x, idx, w):
    """``y[o, m] = sum_t w[o, t] * x[idx[o, t], m]``."""
    return jnp.einsum("ok,okm->om", w, x[idx], precision=jax.lax.Precision.HIGHEST)


@jax.custom_vjp
def _resize_axis(x, idx, w, idx_t, w_t):
    """Banded resize along axis 0; the adjoint is a banded gather, not a scatter."""
    return _gather(x, idx, w)


def _resize_axis_fwd(x, idx, w, idx_t, w_t):
    return _gather(x, idx, w), (idx_t, w_t)


def _resize_axis_bwd(res, g):
    idx_t, w_t = res
    return (_gather(g, idx_t, w_t), None, None, None, None)


_resize_axis.defvjp(_resize_axis_fwd, _resize_axis_bwd)


def banded_scale_and_translate(
    x, out_shape, scale, translation, method: str = "linear"
):
    """Banded equivalent of ``jax.image.scale_and_translate`` with antialiasing.

    Resamples the first two axes of ``x``; any trailing axes are batch. Agrees
    with ``jax.image.scale_and_translate`` to float32 accumulation noise (and
    exactly, for kernels/scales where the tap sets coincide).

    Args:
        x: Input array with the two resampled axes first.
        out_shape: Output size of the two resampled axes.
        scale: Per-axis scale, as ``jax.image`` defines it.
        translation: Per-axis translation, as ``jax.image`` defines it.
        method: One of ``KERNELS`` (``"linear"``, ``"cubic"``, ``"lanczos3"``,
            ``"lanczos5"``).
    """
    if method not in KERNELS:
        raise ValueError(f"Unknown method {method!r}; expected one of {[*KERNELS]}")
    kernel, radius = KERNELS[method]
    scale = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(scale, jnp.float32)), (2,))
    translation = jnp.broadcast_to(
        jnp.atleast_1d(jnp.asarray(translation, jnp.float32)), (2,)
    )
    # Resize axis 0, then swap axes 0<->1 and repeat; two swaps restore the
    # original order: (h,w,..) -> (o0,w,..) -> (w,o0,..) -> (o1,o0,..) -> (o0,o1,..)
    for axis in (0, 1):
        in_size, out_size = x.shape[0], out_shape[axis]
        w_fwd = _forward_width(radius, in_size, out_size)
        w_tr = _transpose_width(radius, in_size, out_size)
        common = (in_size, out_size, scale[axis], translation[axis], kernel, radius)
        idx, weights = _forward_band(*common, w_fwd)
        idx_t, weights_t = _transpose_band(*common, w_tr, w_fwd)
        rest = x.shape[1:]
        flat = _resize_axis(x.reshape(in_size, -1), idx, weights, idx_t, weights_t)
        x = jnp.swapaxes(flat.reshape((out_size, *rest)), 0, 1)
    return x
