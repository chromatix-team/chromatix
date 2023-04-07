# %%
import jax.numpy as jnp
from chromatix import ScalarField, VectorField
import chromatix.functional as cf
from functools import partial
from chex import assert_shape
import pytest


@pytest.mark.parametrize(
    "power, shape, pupil",
    [
        (1.0, (512, 512), partial(cf.circular_pupil, w=10.0)),
        (100.0, (256, 256), None),
    ],
)
def test_plane_wave(power, shape, pupil):
    field = ScalarField.create(0.1, 0.532, 1.0, shape=shape)
    field = cf.plane_wave(field, power, pupil=pupil)

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *shape, 1, 1))


@pytest.mark.parametrize(
    "power, amplitude, shape, pupil",
    [
        (1.0, cf.linear(jnp.pi / 2), (512, 512), partial(cf.circular_pupil, w=10.0)),
        (100.0, cf.left_circular(), (256, 256), None),
    ],
)
def test_plane_wave_vectorial(power, amplitude, shape, pupil):
    field = VectorField.create(0.1, 0.532, 1.0, shape=shape)
    field = cf.plane_wave(field, amplitude=amplitude, power=power, pupil=pupil)

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *shape, 1, 3))


@pytest.mark.parametrize(
    "power, z, shape, pupil",
    [
        (1.0, 0.5, (512, 512), partial(cf.square_pupil, w=10.0)),
        (100.0, 1.0, (256, 256), None),
    ],
)
def test_point_source(power, z, shape, pupil):
    field = ScalarField.create(0.1, 0.532, 1.0, shape=shape)
    field = cf.point_source(field, z, 1.33, power, pupil)

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *shape, 1, 1))


@pytest.mark.parametrize(
    "power, z, shape", [(1.0, 0.5, (512, 512)), (100.0, 1.0, (256, 256))]
)
def test_objective_point_source(power, z, shape):
    field = ScalarField.create(0.1, 0.532, 1.0, shape=shape)
    field = cf.objective_point_source(field, z, 100.0, 1.33, NA=0.8, power=power)

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *shape, 1, 1))
