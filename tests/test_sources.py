from functools import partial

import jax.numpy as jnp
import pytest
from chex import assert_shape

import chromatix.functional as cf


@pytest.mark.parametrize(
    "power, shape, pupil",
    [
        (1.0, (512, 512), partial(cf.circular_pupil, w=10.0)),
        (100.0, (256, 256), None),
    ],
)
def test_plane_wave(power, shape, pupil):
    field = cf.plane_wave(shape, 0.1, 0.532, 1.0, power=power, pupil=pupil)

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
    field = cf.plane_wave(
        shape,
        0.1,
        0.532,
        1.0,
        power=power,
        amplitude=amplitude,
        pupil=pupil,
        scalar=False,
    )

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *shape, 1, 3))


def test_spectral_plane_wave():
    """Tests the planewave initialisation shapes."""
    field = cf.plane_wave(
        (16, 16),
        0.1,
        [0.1, 0.532, 1.0],
        [1.0, 1.0, 1.0],
    )
    assert_shape(field.u, (1, 16, 16, 3, 1))


@pytest.mark.parametrize(
    "power, z, shape, pupil",
    [
        (1.0, 0.5, (512, 512), partial(cf.square_pupil, w=10.0)),
        (100.0, 1.0, (256, 256), None),
    ],
)
def test_point_source(power, z, shape, pupil):
    field = cf.point_source(shape, 0.1, 0.532, 1.0, z, 1.33, power=power, pupil=pupil)

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *shape, 1, 1))


@pytest.mark.parametrize(
    "power, z, shape", [(1.0, 0.5, (512, 512)), (100.0, 1.0, (256, 256))]
)
def test_objective_point_source(power, z, shape):
    field = cf.objective_point_source(
        shape, 0.1, 0.532, 1.0, z, 100.0, 1.33, NA=0.8, power=power
    )

    assert jnp.allclose(field.power, power)
    assert_shape(field.u, (1, *shape, 1, 1))
