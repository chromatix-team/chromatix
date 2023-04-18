import jax.numpy as jnp
from chromatix.functional.samples import multislice_thick_sample, thin_sample
from chromatix.functional.sources import plane_wave


def test_zero_thin_sample():
    # all zero sample, no effect on incoming field expected
    field = plane_wave(
        shape=(2, 2), dx=0.1, spectrum=0.532, spectral_density=1.0, power=1.0
    )
    absorption = jnp.zeros_like(field.u)
    dn = jnp.zeros_like(field.u)
    out_field = thin_sample(field, absorption, dn, thickness=1)
    assert jnp.allclose(field.u, out_field.u)


def test_phase_delay_thin_sample():
    # pure phase sample, no power difference expected
    field = plane_wave(
        shape=(2, 2), dx=0.1, spectrum=0.532, spectral_density=1.0, power=1.0
    )
    absorption = jnp.zeros_like(field.u)
    dn = jnp.ones_like(field.u) * 0.5  # half cycle delay
    out_field = thin_sample(field, absorption, dn, thickness=0.532)
    assert jnp.allclose(field.power, out_field.power)
    assert jnp.allclose(field.u, -out_field.u)


def test_absorption_only_thin_sample():
    # pure absorption sample, no phase difference expected
    field = plane_wave(
        shape=(2, 2), dx=0.1, spectrum=0.532, spectral_density=1.0, power=1.0
    )
    absorption = jnp.ones_like(field.u) / (
        2 * jnp.pi
    )  # gives attenuation factor of 1/e
    dn = jnp.zeros_like(field.u)
    out_field = thin_sample(field, absorption, dn, thickness=0.532)
    assert jnp.allclose(field.power, out_field.power * jnp.exp(2))
    assert jnp.allclose(field.u, out_field.u * jnp.e)


def test_zero_thick_sample():
    # all zero sample, no effect on incoming field expected
    field = plane_wave(
        shape=(2, 2), dx=0.1, spectrum=0.532, spectral_density=1.0, power=1.0
    )
    absorption = jnp.zeros((4, 2, 2))  # D, H, W
    dn = jnp.zeros_like(absorption)
    out_field = multislice_thick_sample(
        field=field,
        absorption_stack=absorption,
        dn_stack=dn,
        n=1.33,
        thickness_per_slice=1.0,
        N_pad=0,
    )
    assert jnp.allclose(field.u, out_field.u)


def test_absorption_only_thick_sample():
    # pure absorption sample, no phase difference expected
    field = plane_wave(
        shape=(2, 2), dx=0.1, spectrum=0.532, spectral_density=1.0, power=1.0
    )
    absorption = jnp.ones((4, 2, 2)) / (2 * jnp.pi)  # gives attenuation factor of 1/e
    dn = jnp.zeros_like(absorption)
    out_field = multislice_thick_sample(
        field=field,
        absorption_stack=absorption,
        dn_stack=dn,
        n=1.33,
        thickness_per_slice=0.532,
        N_pad=0,
    )
    assert jnp.allclose(field.u, out_field.u * jnp.exp(4))


def test_phase_delay_thick_sample():
    # pure phase sample, no power difference expected
    field = plane_wave(
        shape=(2, 2), dx=0.1, spectrum=0.532, spectral_density=1.0, power=1.0
    )
    absorption = jnp.zeros((4, 2, 2))
    dn = jnp.ones_like(absorption) * 0.5  # half cycle delay
    out_field = multislice_thick_sample(
        field=field,
        absorption_stack=absorption,
        dn_stack=dn,
        n=1.33,
        thickness_per_slice=0.532,
        N_pad=0,
    )
    assert jnp.allclose(field.power, out_field.power)
    assert jnp.allclose(field.u, out_field.u)