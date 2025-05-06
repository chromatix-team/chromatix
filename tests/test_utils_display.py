"""
Tests copied from [this file in MacroMax](https://github.com/tttom/MacroMax/blob/master/python/tests/test_utils.py) and
adapted for JAX.
"""
import unittest

import jax.numpy as jnp
import numpy.testing as npt

from chromatix.utils.display import complex2rgb, grid2extent, hsv2rgb


class TestUtilsDisplay(unittest.TestCase):
    def test_complex2rgb(self):
        c = jnp.asarray([[1, jnp.exp(2j*jnp.pi/3)], [jnp.exp(-2j*jnp.pi/3), -1]], dtype=jnp.complex128)
        res = jnp.asarray([[[0, 1, 1], [1, 0, 1]], [[1, 1, 0], [1, 0, 0]]], dtype=jnp.float64)

        rgb = complex2rgb(c)
        npt.assert_array_almost_equal(rgb, res)

        # Check saturation
        rgb = complex2rgb(10.0 * c)
        npt.assert_array_almost_equal(rgb, jnp.ones_like(res), err_msg='Saturated values are not as expected.')

        # Check intensity scaling
        rgb = complex2rgb(0.5 * c)
        npt.assert_array_almost_equal(rgb, 0.5 * res, err_msg='The intensity does not scale linearly with the amplitude.')

        c = c.at[1, 1].divide(2)
        res = res.at[1, 1].divide(2)
        rgb = complex2rgb(0.5 * c)
        npt.assert_array_almost_equal(rgb, 0.5 * res, err_msg='Non-uniform amplitudes are not represented correctly.')

    def test_hsv2rgb(self):
        hsv = jnp.asarray([[[0, 1, 1], [1.0/3.0, 1, 1]], [[2.0/3.0, 1, 1], [1, 1, 0.5]]])
        res = jnp.asarray([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0.5, 0, 0]]])

        rgb = hsv2rgb(hsv)
        npt.assert_array_almost_equal(rgb, res, err_msg='HSV to RGB hue test failed')

    def test_ranges2extent(self):
        npt.assert_array_almost_equal(grid2extent(jnp.arange(5), jnp.arange(10)), [-0.5, 9.5, 4.5, -0.5])
        npt.assert_array_almost_equal(grid2extent(jnp.arange(-2, 5 - 2), jnp.arange(-3, 10 - 3)), [-3.5, 6.5, 2.5, -2.5])
        npt.assert_array_almost_equal(grid2extent(jnp.arange(5), jnp.arange(0, 2, 0.2)), [-0.1, 1.9, 4.5, -0.5])
        npt.assert_array_almost_equal(grid2extent(jnp.arange(5), jnp.arange(-1, 2, 0.2)), [-1.1, 1.9, 4.5, -0.5])
        npt.assert_array_almost_equal(grid2extent(range(-2, 5 - 2), jnp.arange(-3, 10 - 3)), [-3.5, 6.5, 2.5, -2.5])
        npt.assert_array_almost_equal(grid2extent(range(-2, 5 - 2), range(-3, 10 - 3)), [-3.5, 6.5, 2.5, -2.5])
        npt.assert_array_almost_equal(grid2extent(jnp.arange(-2, 5 - 2), range(-3, 10 - 3)), [-3.5, 6.5, 2.5, -2.5])


if __name__ == '__main__':
    unittest.main()
