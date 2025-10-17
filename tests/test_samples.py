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
