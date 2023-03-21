# %%
import jax.numpy as jnp
import chromatix.functional as cf
from functools import partial
from chex import assert_shape
import pytest


@pytest.mark.parametrize(
    "power, phase, size, pupil",
    [
        (1.0, jnp.pi, (512, 512), partial(cf.circular_pupil, w=10.0)),
        (100.0, -jnp.pi, (256, 256), None),
    ],
)
def test_plane_wave(power, phase, size, pupil):
    field = cf.empty_field(size, 0.1, 0.532, 1.0)
    field = cf.plane_wave(field, power, pupil=pupil)

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *size, 1))


@pytest.mark.parametrize(
    "power, z, size, pupil",
    [
        (1.0, 0.5, (512, 512), partial(cf.square_pupil, w=10.0)),
        (100.0, 1.0, (256, 256), None),
    ],
)
def test_point_source(power, z, size, pupil):
    field = cf.empty_field(size, 0.1, 0.532, 1.0)
    field = cf.point_source(field, z, 1.33, power, pupil)

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *size, 1))


@pytest.mark.parametrize(
    "power, z, size", [(1.0, 0.5, (512, 512)), (100.0, 1.0, (256, 256))]
)
def test_objective_point_source(power, z, size):
    field = cf.empty_field(size, 0.1, 0.532, 1.0)
    field = cf.objective_point_source(field, z, 100.0, 1.33, NA=0.8, power=power)

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *size, 1))
