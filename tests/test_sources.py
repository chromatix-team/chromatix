from functools import partial

from jax import Array

import chromatix.functional as cf
import jax.numpy as jnp
import pytest
from chex import assert_shape


class TestPlaneWave:
    @pytest.mark.parametrize(
        "power, shape, pupil",
        [
            (1.0, (512, 512), partial(cf.circular_pupil, w=10.0)),
            (100.0, (256, 256), None),
        ],
    )
    def test_scalar(self, power, shape, pupil):
        field = cf.plane_wave(shape, 0.1, 0.532, 1.0, power=power, pupil=pupil)

        assert jnp.allclose(field.power, power)
        assert_shape(field.u, (1, *shape, 1, 1))

    @pytest.mark.parametrize(
        "power, amplitude, shape, pupil",
        [
            (
                1.0,
                cf.linear(jnp.pi / 2),
                (512, 512),
                partial(cf.circular_pupil, w=10.0),
            ),
            (100.0, cf.left_circular(), (256, 256), None),
        ],
    )
    def test_vectorial(self, power, amplitude, shape, pupil):
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
        "amplitude, scalar",
        [
            (jnp.ones((2,)), True),
            (jnp.ones((2,)), False),
            (jnp.ones((1,)), False),
            (jnp.ones((3,)), True),
        ],
    )
    def test_wrong_amplitude_size(self, amplitude: Array, scalar: bool):
        # Using wrong shape for amplitude should raise Assertion error
        with pytest.raises(AssertionError):
            _ = cf.plane_wave(
                (256, 256),
                0.1,
                0.532,
                1.0,
                amplitude=amplitude,
                scalar=scalar,
                power=1.0,
            )


class TestObjectivePointSource:
    @pytest.mark.parametrize(
        "power, z, shape",
        [(1.0, 0.5, (512, 512)), (100.0, 1.0, (256, 256)), (50, 0.0, (256, 512))],
    )
    def test_scalar(self, power, z, shape):
        field = cf.objective_point_source(
            shape, 0.1, 0.532, 1.0, z, 100.0, 1.33, NA=0.8, power=power
        )

        assert jnp.allclose(field.power, power)
        assert_shape(field.u, (1, *shape, 1, 1))

    def test_vectorial(self):
        field = cf.point_source(
            (512, 512),
            0.1,
            0.532,
            1.0,
            1.0,
            amplitude=cf.left_circular(),
            n=1.33,
            power=1.0,
            pupil=None,
            scalar=False,
        )
        assert jnp.allclose(field.power, 1.0)
        assert_shape(field.u, (1, 512, 512, 1, 3))

    @pytest.mark.parametrize(
        "amplitude, scalar",
        [
            (jnp.ones((2,)), True),
            (jnp.ones((2,)), False),
            (jnp.ones((1,)), False),
            (jnp.ones((3,)), True),
        ],
    )
    def test_wrong_amplitude_size(self, amplitude: Array, scalar: bool):
        # Using wrong shape for amplitude should raise Assertion error
        with pytest.raises(AssertionError):
            _ = cf.objective_point_source(
                (256, 256),
                0.1,
                0.532,
                1.0,
                1.0,
                100,
                1.0,
                1.33,
                amplitude=amplitude,
                scalar=scalar,
                power=1.0,
            )


class TestPointSource:
    @pytest.mark.parametrize(
        "power, z, shape, pupil",
        [
            (1.0, 0.5, (512, 512), partial(cf.square_pupil, w=10.0)),
            (100.0, 1.0, (256, 256), None),
            (20.0, 2, (256, 512), None),  # should work for z =0
        ],
    )
    def test_scalar(self, power, z, shape, pupil):
        field = cf.point_source(
            shape, 0.1, 0.532, 1.0, z, 1.33, power=power, pupil=pupil
        )

        assert jnp.allclose(field.power, power)
        assert_shape(field.u, (1, *shape, 1, 1))

    def test_vectorial(self):
        field = cf.point_source(
            (512, 512),
            0.1,
            0.532,
            1.0,
            1.0,
            amplitude=cf.left_circular(),
            n=1.33,
            power=1.0,
            pupil=None,
            scalar=False,
        )
        assert jnp.allclose(field.power, 1.0)
        assert_shape(field.u, (1, 512, 512, 1, 3))

    @pytest.mark.parametrize(
        "amplitude, scalar",
        [
            (jnp.ones((2,)), True),
            (jnp.ones((2,)), False),
            (jnp.ones((1,)), False),
            (jnp.ones((3,)), True),
        ],
    )
    def test_wrong_amplitude_size(self, amplitude: Array, scalar: bool):
        # Using wrong shape for amplitude should raise Assertion error
        with pytest.raises(AssertionError):
            _ = cf.point_source(
                (256, 256),
                0.1,
                0.532,
                1.0,
                1.0,
                1.33,
                amplitude=amplitude,
                scalar=scalar,
                power=1.0,
                pupil=None,
            )
