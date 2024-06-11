from functools import partial

import chromatix.functional as cf
import jax.numpy as jnp
import pytest
from chex import assert_shape


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


def test_point_source_wrong_size():
    with pytest.raises(AssertionError) as e_info:
        field = cf.point_source(
            (256, 256),
            0.1,
            0.532,
            1.0,
            1.0,
            1.33,
            amplitude=jnp.array([1.0, 1.0]),
            scalar=True,
            power=1.0,
            pupil=None,
        )
    with pytest.raises(AssertionError) as e_info:
        field = cf.point_source(
            (256, 256),
            0.1,
            0.532,
            1.0,
            1.0,
            1.33,
            amplitude=jnp.array([1.0, 1.0]),
            scalar=False,
            power=1.0,
            pupil=None,
        )

    with pytest.raises(AssertionError) as e_info:
        field = cf.point_source(
            (256, 256),
            0.1,
            0.532,
            1.0,
            1.0,
            1.33,
            amplitude=jnp.array([1.0, 1.0, 1.0]),
            scalar=True,
            power=1.0,
            pupil=None,
        )

    with pytest.raises(AssertionError) as e_info:
        field = cf.point_source(
            (256, 256),
            0.1,
            0.532,
            1.0,
            1.0,
            1.33,
            amplitude=1.0,
            scalar=False,
            power=1.0,
            pupil=None,
        )
