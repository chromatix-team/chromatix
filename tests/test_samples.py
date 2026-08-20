import jax.numpy as jnp
import pytest

from chromatix.elements.samples import MultisliceThickSample, ThinSample
from chromatix.functional.samples import multislice_thick_sample, thin_sample
from chromatix.functional.sources import plane_wave


def test_zero_thin_sample():
    # All zero sample, no effect on incoming field expected
    field = plane_wave(shape=(2, 2), dx=0.1, spectrum=(0.532, 1.0), power=1.0)
    absorption = jnp.zeros(field.spatial_shape)
    dn = jnp.zeros(field.spatial_shape)
    out_field = thin_sample(field, absorption, dn, thickness=1.0)
    assert jnp.allclose(field.u, out_field.u)
    sample = ThinSample(absorption, dn, dx=0.1, thickness=1.0)
    sample_out_field = sample(field)
    assert jnp.allclose(sample_out_field.u, out_field.u)


def test_phase_delay_thin_sample():
    # Pure phase sample, no power difference expected
    field = plane_wave(shape=(2, 2), dx=0.1, spectrum=(0.532, 1.0), power=1.0)
    absorption = jnp.zeros(field.spatial_shape)
    dn = jnp.ones(field.spatial_shape) * 0.5  # Half cycle delay
    out_field = thin_sample(field, absorption, dn, thickness=0.532)
    assert jnp.allclose(field.power, out_field.power)
    assert jnp.allclose(field.u, -out_field.u)
    sample = ThinSample(absorption, dn, dx=0.1, thickness=0.532)
    sample_out_field = sample(field)
    assert jnp.allclose(sample_out_field.u, out_field.u)


def test_absorption_only_thin_sample():
    # pure absorption sample, no phase difference expected
    field = plane_wave(shape=(2, 2), dx=0.1, spectrum=(0.532, 1.0), power=1.0)
    absorption = jnp.ones(field.spatial_shape) / (
        2 * jnp.pi
    )  # Gives attenuation factor of 1/e
    dn = jnp.zeros(field.spatial_shape)
    out_field = thin_sample(field, absorption, dn, thickness=0.532)
    assert jnp.allclose(field.power, out_field.power * jnp.exp(2))
    assert jnp.allclose(field.u, out_field.u * jnp.e)
    sample = ThinSample(absorption, dn, dx=0.1, thickness=0.532)
    sample_out_field = sample(field)
    assert jnp.allclose(sample_out_field.u, out_field.u)


def test_zero_thick_sample():
    # All zero sample, no effect on incoming field expected
    field = plane_wave(shape=(2, 2), dx=0.1, spectrum=(0.532, 1.0), power=1.0)
    absorption = jnp.zeros((4, 2, 2))
    dn = jnp.zeros_like(absorption)
    out_field = multislice_thick_sample(
        field=field,
        absorption_stack=absorption,
        dn_stack=dn,
        n=1.33,
        thickness_per_slice=1.0,
        pad_width=0,
    )
    assert jnp.allclose(field.u, out_field.u)
    sample = MultisliceThickSample(absorption, dn, n=1.33, dx=0.1, thickness=1.0)
    sample_out_field = sample(field)
    assert jnp.allclose(sample_out_field.u, out_field.u)


@pytest.mark.skip("The math doesn't make sense here.")
def test_absorption_only_thick_sample():
    # pure absorption sample, no phase difference expected
    field = plane_wave(shape=(2, 2), dx=0.1, spectrum=(0.532, 1.0), power=1.0)
    absorption = jnp.ones((4, 2, 2)) / (2 * jnp.pi)  # gives attenuation factor of 1/e
    dn = jnp.zeros_like(absorption)
    out_field = multislice_thick_sample(
        field=field,
        absorption_stack=absorption,
        dn_stack=dn,
        n=1.33,
        thickness_per_slice=0.532,
        pad_width=0,
    )
    assert jnp.allclose(field.u, out_field.u * jnp.exp(4))


def test_phase_delay_thick_sample():
    # Pure phase sample, no power difference expected
    field = plane_wave(shape=(2, 2), dx=0.1, spectrum=(0.532, 1.0), power=1.0)
    absorption = jnp.zeros((4, 2, 2))
    dn = jnp.ones_like(absorption) * 0.5  # Half cycle delay
    out_field = multislice_thick_sample(
        field=field,
        absorption_stack=absorption,
        dn_stack=dn,
        n=1.33,
        thickness_per_slice=0.532,
        pad_width=0,
    )
    assert jnp.allclose(field.power, out_field.power)
    sample = MultisliceThickSample(absorption, dn, n=1.33, dx=0.1, thickness=0.532)
    sample_out_field = sample(field)
    assert jnp.allclose(sample_out_field.u, out_field.u)


# NOTE: every test above uses ``pad_width=0``, so the padded path -- where the
# sample stacks are smaller than the (padded) field -- was previously uncovered.
# The tests below exercise it.


def _multislice_padded_stacks_reference(
    field, absorption, dn, n, thickness, pad_width, reverse_propagate_distance=None
):
    """Pre-2026-08 semantics: zero-pad the stacks up to the padded field shape.

    Written out explicitly (rather than by calling the function under test) so
    the centre-application fast path is pinned to the behaviour it replaced.
    """
    import jax

    from chromatix import crop
    from chromatix import pad as pad_field
    from chromatix.functional.propagation import (
        compute_asm_propagator,
        kernel_propagate,
    )
    from chromatix.utils import center_pad

    depth = dn.shape[0]
    field = pad_field(field, pad_width)
    absorption = center_pad(absorption, (0, pad_width, pad_width))
    dn = center_pad(dn, (0, pad_width, pad_width))
    propagator = compute_asm_propagator(field, thickness, n, (0.0, 0.0))

    def step(i, u):
        field_i = kernel_propagate(field.replace(u=u), propagator)
        return thin_sample(field_i, absorption[i], dn[i], thickness).u

    field = field.replace(u=jax.lax.fori_loop(0, depth, step, field.u))
    if reverse_propagate_distance is None:
        reverse_propagate_distance = thickness * depth / 2
    field = kernel_propagate(
        field, compute_asm_propagator(field, -reverse_propagate_distance, n)
    )
    return crop(field, pad_width).u


@pytest.mark.parametrize(
    ("shape", "depth", "pad_width"), [((8, 8), 3, 4), ((6, 6), 2, 3), ((8, 8), 4, 2)]
)
def test_thick_sample_padding_matches_padded_stacks(shape, depth, pad_width):
    # The sample occupies only the unpadded centre, so applying it there must be
    # identical to zero-padding the stacks (the ring transmission is exp(0) = 1).
    import jax

    field = plane_wave(shape=shape, dx=0.1, spectrum=(0.532, 1.0), power=1.0)
    dn = 1e-3 * jax.random.normal(jax.random.key(0), (depth, *shape))
    absorption = 1e-4 * jnp.abs(jax.random.normal(jax.random.key(1), (depth, *shape)))
    for reverse in (None, 0.25):
        out_field = multislice_thick_sample(
            field=field,
            absorption_stack=absorption,
            dn_stack=dn,
            n=1.33,
            thickness_per_slice=0.532,
            pad_width=pad_width,
            reverse_propagate_distance=reverse,
        )
        expected = _multislice_padded_stacks_reference(
            field, absorption, dn, 1.33, 0.532, pad_width, reverse
        )
        assert out_field.u.shape == expected.shape
        assert jnp.allclose(out_field.u, expected, rtol=0, atol=0), (
            f"not bit-identical for reverse={reverse}"
        )


def test_thick_sample_padding_is_lossy_but_finite():
    # Padding the field with zeros and cropping back discards the light that
    # diffracts into the ring, so power is NOT conserved here -- assert only that
    # the padded path stays finite and loses (never gains) power.
    field = plane_wave(shape=(8, 8), dx=0.1, spectrum=(0.532, 1.0), power=1.0)
    absorption = jnp.zeros((3, 8, 8))
    dn = jnp.ones_like(absorption) * 0.5
    out_field = multislice_thick_sample(
        field=field,
        absorption_stack=absorption,
        dn_stack=dn,
        n=1.33,
        thickness_per_slice=0.532,
        pad_width=4,
    )
    assert jnp.isfinite(out_field.u).all()
    assert out_field.power <= field.power * (1.0 + 1e-5)


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
def test_thick_sample_padding_supports_batched_fields(batch_shape):
    # Regression: the centre indexer must target the SPATIAL axes. A bare
    # 2-tuple of slices hits the leading axes instead, which works for an
    # unbatched field and raises for any batched one.
    import jax

    shape, depth, pad_width = (8, 8), 3, 4
    field = plane_wave(shape=shape, dx=0.1, spectrum=(0.532, 1.0), power=1.0)
    field = field.replace(u=jnp.broadcast_to(field.u, batch_shape + field.u.shape))
    dn = 1e-3 * jax.random.normal(jax.random.key(0), (depth, *shape))
    absorption = 1e-4 * jnp.abs(jax.random.normal(jax.random.key(1), (depth, *shape)))
    out_field = multislice_thick_sample(
        field=field,
        absorption_stack=absorption,
        dn_stack=dn,
        n=1.33,
        thickness_per_slice=0.532,
        pad_width=pad_width,
    )
    assert out_field.u.shape == field.u.shape
    assert jnp.isfinite(out_field.u).all()
