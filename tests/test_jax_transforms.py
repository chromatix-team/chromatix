import jax
import jax.numpy as jnp
from jaxtyping import ScalarLike

import chromatix.functional as cx
from chromatix import ChromaticScalarField, Field, MonoSpectrum, ScalarField


def system(z: ScalarLike, wavelength: ScalarLike) -> Field:
    field = cx.plane_wave((256, 256), 40.0 / 256, wavelength)
    field = cx.transform_propagate_sas(
        field, z, 1.33
    )  # NOTE(dd/2025-09-30): A useful test because SAS does not handle multiple z values
    return field


def test_properties_after_vmap_z():
    z = jnp.linspace(-100.0, 100.0, num=3)
    system_vmap_z = jax.vmap(system, in_axes=(0, None))
    field = system_vmap_z(z, 0.532)
    assert field.spectrum.wavelength.shape == (3, 1)
    assert isinstance(field.spectrum, MonoSpectrum)
    assert isinstance(field, ScalarField)
    assert jnp.all(field.spectrum.wavelength == 0.532)
    assert field.shape == (3, 256, 256)


def test_properties_after_vmap_z_multiple_wavelengths():
    z = jnp.linspace(-100.0, 100.0, num=3)
    system_vmap_z = jax.vmap(system, in_axes=(0, None))
    field = system_vmap_z(z, jnp.array([0.532, 0.650]))
    assert field.spectrum.wavelength.shape == (3, 2)
    assert not isinstance(field.spectrum, MonoSpectrum)
    assert isinstance(field, ChromaticScalarField)
    assert field.shape == (3, 256, 256, 2)


def test_properties_after_vmap_z_and_wavelength():
    z = jnp.linspace(-100.0, 100.0, num=3)
    system_vmap_z_and_wavelength = jax.vmap(system, in_axes=(0, 0))
    field = system_vmap_z_and_wavelength(z, jnp.array([0.532, 0.650, 0.670]))
    assert field.spectrum.wavelength.shape == (3, 1)
    assert isinstance(field.spectrum, MonoSpectrum)
    assert jnp.all(
        field.spectrum.wavelength.squeeze() == jnp.array([0.532, 0.650, 0.670])
    )
    assert field.shape == (3, 256, 256)


def test_properties_after_vmap_z_and_multiple_wavelengths():
    z = jnp.linspace(-100.0, 100.0, num=3)
    system_vmap_z_and_multiple_wavelengths = jax.vmap(system, in_axes=(0, 0))
    field = system_vmap_z_and_multiple_wavelengths(
        z, jnp.array([[0.532, 0.650], [0.432, 0.550], [0.632, 0.450]])
    )
    assert field.spectrum.wavelength.shape == (3, 2)
    assert not isinstance(field.spectrum, MonoSpectrum)
    assert jnp.all(
        field.spectrum.wavelength
        == jnp.array([[0.532, 0.650], [0.432, 0.550], [0.632, 0.450]])
    )
    assert field.shape == (3, 256, 256, 2)


def test_properties_after_jit_vmap_z():
    z = jnp.linspace(-100.0, 100.0, num=3)
    system_jit_vmap_z = jax.jit(jax.vmap(system, in_axes=(0, None)))
    field = system_jit_vmap_z(z, 0.532)
    assert field.spectrum.wavelength.shape == (3, 1)
    assert isinstance(field.spectrum, MonoSpectrum)
    assert isinstance(field, ScalarField)
    assert field.shape == (3, 256, 256)
