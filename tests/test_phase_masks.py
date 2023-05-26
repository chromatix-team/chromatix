from chromatix.functional import (
    plane_wave,
    phase_change,
)
from chromatix.utils import (
    flat_phase,
    defocused_ramps,
    zernike_aberrations,
)

from chromatix.elements import PhaseMask, trainable
import jax
import jax.numpy as jnp


def test_flat_phase():
    shape = (256, 256)
    field = plane_wave(shape, dx=0.3, spectrum=0.532, spectral_density=1.0)
    phase = flat_phase(shape)
    field_after_phase_mask = phase_change(field, phase)
    # There should be no change to the intensity
    assert jnp.allclose(
        field.intensity,
        field_after_phase_mask.intensity,
    )
    # There should be no change to the phase
    assert jnp.allclose(
        field.phase,
        field_after_phase_mask.phase,
    )


def test_defocused_ramps():
    shape = (256, 256)
    spacing = 0.3
    f = 100.0
    n = 1.33
    NA = 0.5
    wavelength = 0.532
    field = plane_wave(shape, dx=spacing, spectrum=wavelength, spectral_density=1.0)
    phase = defocused_ramps(shape, spacing, wavelength, n, f, NA)
    field_after_phase_mask = phase_change(field, phase)
    # There should be no change to the intensity
    assert jnp.allclose(
        field.intensity,
        field_after_phase_mask.intensity,
    )
    # Phase change should match expected value
    assert jnp.allclose(
        jnp.angle(field.u.squeeze() * jnp.exp(1j * phase)),
        field_after_phase_mask.phase.squeeze(),
    )


def test_zernike_aberrations():
    shape = (256, 256)
    spacing = 0.3
    f = 100.0
    n = 1.33
    NA = 0.5
    wavelength = 0.532
    field = plane_wave(shape, dx=spacing, spectrum=wavelength, spectral_density=1.0)
    phase = zernike_aberrations(
        shape,
        spacing,
        wavelength,
        n,
        f,
        NA,
        ansi_indices=[0, 1, 2, 3, 4],
        coefficients=[0, 0, 0, 0, 5],
    )
    field_after_phase_mask = phase_change(field, phase)
    # There should be no change to the intensity
    assert jnp.allclose(
        field.intensity,
        field_after_phase_mask.intensity,
    )
    # Phase change should match expected value
    assert jnp.allclose(
        jnp.angle(field.u.squeeze() * jnp.exp(1j * phase)),
        field_after_phase_mask.phase.squeeze(),
    )


def test_phase_mask_element():
    shape = (256, 256)
    spacing = 0.3
    f = 100.0
    n = 1.33
    NA = 0.5
    wavelength = 0.532
    field = plane_wave(shape, dx=spacing, spectrum=wavelength, spectral_density=1.0)
    model = PhaseMask(trainable(defocused_ramps, rng=False), f, n, NA)
    variables = model.init(jax.random.PRNGKey(4), field)
    field_after_phase_mask = model.apply(variables, field)
    phase = defocused_ramps(shape, spacing, wavelength, n, f, NA)
    field_after_phase_mask_function = phase_change(field, phase)
    # There should be no change to the intensity
    assert jnp.allclose(
        field.intensity,
        field_after_phase_mask.intensity,
    )
    # Phase change should match expected value
    assert jnp.allclose(
        field_after_phase_mask_function.phase,
        field_after_phase_mask.phase,
    )
