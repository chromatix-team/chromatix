"""Resampling: the banded path must agree with jax.image, values and gradients."""

import jax
import jax.numpy as jnp
import pytest
from jax.image import scale_and_translate

from chromatix.ops import init_plane_resample

METHODS = ["linear", "cubic", "lanczos3", "lanczos5"]
# (in, out): same size, 2x/4x down, and 2x UP -- upsampling exercises the
# transpose band, which is the wide one in that direction. Sizing it from the
# forward width silently truncates the adjoint: values stay right while
# gradients are wrong, so the gradient check below is the one that matters.
SHAPES = [(64, 64), (64, 32), (64, 16), (32, 64), (48, 24)]


def _resamplers(out_shape, out_spacing, method):
    return (
        init_plane_resample(out_shape, out_spacing, method, banded=True),
        init_plane_resample(out_shape, out_spacing, method, banded=False),
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("n_in", "n_out"), SHAPES)
def test_banded_matches_dense_values(method, n_in, n_out):
    x = jax.random.normal(jax.random.key(0), (n_in, n_in), jnp.float32)
    in_spacing = jnp.array([0.1, 0.1])
    out_spacing = 0.1 * n_in / n_out
    banded, dense = _resamplers((n_out, n_out), out_spacing, method)
    got, expected = banded(x, in_spacing), dense(x, in_spacing)
    assert got.shape == expected.shape
    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("n_in", "n_out"), SHAPES)
def test_banded_matches_dense_gradients(method, n_in, n_out):
    x = jax.random.normal(jax.random.key(1), (n_in, n_in), jnp.float32)
    in_spacing = jnp.array([0.1, 0.1])
    out_spacing = 0.1 * n_in / n_out
    banded, dense = _resamplers((n_out, n_out), out_spacing, method)

    def loss(resampler, v):
        return jnp.sum(resampler(v, in_spacing) ** 2)

    g_banded = jax.grad(lambda v: loss(banded, v))(x)
    g_dense = jax.grad(lambda v: loss(dense, v))(x)
    assert jnp.allclose(g_banded, g_dense, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_banded_handles_trailing_dims_at_the_functional_level(method):
    # The low-level banded helper treats trailing axes as batch. Note that
    # InterpolatingPlaneResampler itself cannot (see the xfail below), so this
    # exercises the helper directly.
    from chromatix.ops._banded_resample import banded_scale_and_translate

    x = jax.random.normal(jax.random.key(2), (64, 64, 3), jnp.float32)
    scale = jnp.array([0.5, 0.5])
    translation = -0.5 * (jnp.array([64.0, 64.0]) * scale - jnp.array([32.0, 32.0]))
    got = banded_scale_and_translate(x, (32, 32), scale, translation, method)
    expected = scale_and_translate(
        x, (32, 32, 3), (0, 1), scale, translation, method=method
    )
    assert got.shape == expected.shape
    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.xfail(
    reason="Pre-existing on main: InterpolatingPlaneResampler computes "
    "`_in_shape * scale` with _in_shape=(ndim,) and scale=(2,), so any input "
    "with trailing dims raises, despite the docstring promising support. "
    "Affects the dense path identically; not introduced by the banded path. "
    "chromatix vmaps over extra dims instead (see elements/sensors.py).",
    raises=TypeError,
    strict=True,
)
def test_resampler_trailing_dims_unsupported():
    x = jax.random.normal(jax.random.key(2), (64, 64, 3), jnp.float32)
    init_plane_resample((32, 32), 0.2, "lanczos3")(x, jnp.array([0.1, 0.1]))


def test_banded_is_jittable_and_vmappable():
    # This is how chromatix actually applies it to arrays with extra dims.
    x = jax.random.normal(jax.random.key(3), (4, 64, 64), jnp.float32)
    in_spacing = jnp.array([0.1, 0.1])
    resampler = init_plane_resample((32, 32), 0.2, "lanczos3")
    batched = jax.jit(jax.vmap(resampler, in_axes=(0, None)))
    assert batched(x, in_spacing).shape == (4, 32, 32)


def test_unknown_method_falls_through_to_jax():
    # Unknown methods bypass the banded path and keep jax.image's own behaviour,
    # including its error message, so this stays backwards compatible.
    with pytest.raises(ValueError, match="Nearest neighbor"):
        init_plane_resample((32, 32), 0.2, "nearest")(
            jnp.zeros((64, 64)), jnp.array([0.1, 0.1])
        )
