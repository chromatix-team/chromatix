import jax.numpy as jnp
import pytest
from chromatix.functional.samples import multislice_thick_sample, thin_sample
from chromatix.functional.sources import empty_field, plane_wave


def test_zero_thin_sample():
    # all zero sample, no effect on incoming field expected
    absorption = jnp.zeros((1, 2, 2, 1))
    dn = jnp.zeros((1, 2, 2, 1))
    field = empty_field((2, 2), 0.1, 0.532, 1.0)
    field = plane_wave(field, power=1)
    out_field = thin_sample(field, absorption, dn, thickness=1)

    assert jnp.allclose(field.u, out_field.u)


def test_phase_delay_thin_sample():
    # pure phase sample, no power difference expected
    absorption = jnp.zeros((1, 2, 2, 1))
    dn = jnp.ones((1, 2, 2, 1)) * 0.5  # half cycle delay
    field = empty_field((2, 2), 0.1, 0.532, 1.0)
    field = plane_wave(field, power=1)
    out_field = thin_sample(field, absorption, dn, thickness=0.532)

    assert jnp.allclose(field.power, out_field.power)
    assert jnp.allclose(field.u, -out_field.u)


def test_absorption_only_thin_sample():
    # pure phase sample, no power difference expected
    absorption = jnp.ones((1, 2, 2, 1)) / (2 * jnp.pi)  # attenuation factor of 1/e
    dn = jnp.zeros((1, 2, 2, 1))
    field = empty_field((2, 2), 0.1, 0.532, 1.0)
    field = plane_wave(field, power=1)
    out_field = thin_sample(field, absorption, dn, thickness=0.532)

    assert jnp.allclose(field.power, out_field.power*jnp.exp(2))
    assert jnp.allclose(field.u, out_field.u*jnp.e)
