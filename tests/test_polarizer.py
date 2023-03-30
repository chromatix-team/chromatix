import jax.numpy as jnp
import chromatix.functional as cf
from chromatix import VectorField
from functools import partial
from chex import assert_axis_dimension


def test_linear_polarizer():
    field = VectorField.create(1.0, 0.532, 1.0, shape=(512, 512))

    k = jnp.array([0.0, 1.0])
    E0 = jnp.ones((3,))

    field = cf.vector_plane_wave(
        field, k, E0, power=2.0, pupil=partial(cf.square_pupil, w=10.0)
    )

    field = cf.linear_polarizer(field, 1 / 2 * jnp.pi)
    assert_axis_dimension(field.u, -1, 3)
    # TODO: add another test


def test_left_circular_polarizer():
    field = VectorField.create(1.0, 0.532, 1.0, shape=(512, 512))

    k = jnp.array([0.0, 1.0])
    E0 = jnp.ones((3,))

    field = cf.vector_plane_wave(
        field, k, E0, power=2.0, pupil=partial(cf.square_pupil, w=10.0)
    )

    field = cf.left_circular_polarizer(field)
    assert_axis_dimension(field.u, -1, 3)
    # TODO: add another test


def test_right_circular_polarizer():
    field = VectorField.create(1.0, 0.532, 1.0, shape=(512, 512))

    k = jnp.array([0.0, 1.0])
    E0 = jnp.ones((3,))

    field = cf.vector_plane_wave(
        field, k, E0, power=2.0, pupil=partial(cf.square_pupil, w=10.0)
    )

    field = cf.right_circular_polarizer(field)
    assert_axis_dimension(field.u, -1, 3)
    # TODO: add another test