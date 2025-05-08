from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from chex import assert_axis_dimension

import chromatix.functional as cf


def test_inits():
    assert jnp.allclose(cf.linear(0), jnp.array([0, 0, 1], dtype=jnp.complex64))
    assert jnp.allclose(
        cf.linear(jnp.pi / 2), jnp.array([0, 1, 0], dtype=jnp.complex64), atol=1e-7
    )
    assert jnp.allclose(cf.left_circular(), jnp.array([0, 1j, 1]) / jnp.sqrt(2))
    assert jnp.allclose(cf.right_circular(), jnp.array([0, -1j, 1]) / jnp.sqrt(2))


@pytest.mark.parametrize(
    ["E0", "angle", "power"],
    [
        (cf.linear(1 / 2 * jnp.pi), 0, 0),
        (cf.linear(1 / 2 * jnp.pi), 1 / 4 * jnp.pi, jnp.cos(1 / 4 * jnp.pi) ** 2),
        (cf.linear(1 / 2 * jnp.pi), 1 / 2 * jnp.pi, 1),
        (cf.linear(0), 0, 1),
        (cf.linear(0), 1 / 4 * jnp.pi, jnp.cos(1 / 4 * jnp.pi) ** 2),
        (cf.linear(0), 1 / 2 * jnp.pi, 0),
    ],
)
def test_linear_polarizer(E0, angle, power):
    field = cf.plane_wave(
        (512, 512),
        1.0,
        0.532,
        1.0,
        amplitude=E0,
        power=1.0,
        pupil=partial(cf.square_pupil, w=10.0),
        scalar=False,
    )
    field = cf.linear_polarizer(field, angle=angle)

    # check shape
    assert_axis_dimension(field.u, -1, 3)

    # check power - malus law
    assert jnp.allclose(field.power.squeeze(), power, atol=1e-3, rtol=1e-3)


def test_left_circular_polarizer():
    # right circular light through left ciruclar
    # polariser shouldnt give any output
    field = cf.plane_wave(
        (512, 512),
        1.0,
        0.532,
        1.0,
        amplitude=cf.right_circular(),
        power=1.0,
        pupil=partial(cf.square_pupil, w=10.0),
        scalar=False,
    )
    field = cf.left_circular_polarizer(field)
    assert_axis_dimension(field.u, -1, 3)
    assert np.allclose(field.power, 0.0)

    # Right circular light through right ciruclar
    # polariser should return right circular light
    field = cf.plane_wave(
        (512, 512),
        1.0,
        0.532,
        1.0,
        amplitude=cf.left_circular(),
        power=1.0,
        pupil=partial(cf.square_pupil, w=10.0),
        scalar=False,
    )
    field_after = cf.left_circular_polarizer(field)
    assert_axis_dimension(field.u, -1, 3)
    assert np.allclose(field.u, field_after.u, atol=5e-5, rtol=5e-5)


def test_right_circular_polarizer():
    # Left circular light through right ciruclar
    # polariser shouldnt give any output
    field = cf.plane_wave(
        (512, 512),
        1.0,
        0.532,
        1.0,
        amplitude=cf.left_circular(),
        power=1.0,
        pupil=partial(cf.square_pupil, w=10.0),
        scalar=False,
    )
    field = cf.right_circular_polarizer(field)
    assert_axis_dimension(field.u, -1, 3)
    assert np.allclose(field.power, 0.0)

    # Right circular light through right ciruclar
    # polariser should return right circular light
    field = cf.plane_wave(
        (512, 512),
        1.0,
        0.532,
        1.0,
        amplitude=cf.right_circular(),
        power=1.0,
        pupil=partial(cf.square_pupil, w=10.0),
        scalar=False,
    )
    field_after = cf.right_circular_polarizer(field)
    assert_axis_dimension(field.u, -1, 3)
    assert np.allclose(field.u, field_after.u, atol=5e-5, rtol=5e-5)


def test_quarter_waveplate():
    field = cf.plane_wave(
        (512, 512),
        1.0,
        0.532,
        1.0,
        amplitude=cf.linear(0),
        power=1.0,
        pupil=partial(cf.square_pupil, w=10.0),
        scalar=False,
    )

    # Linear with quarterwave at pi/4 yields right circular.
    field = cf.quarterwave_plate(field, jnp.pi / 4)
    assert jnp.allclose(
        field.jones_vector[0, 256, 256, 0], cf.right_circular(), atol=1e-7
    )
