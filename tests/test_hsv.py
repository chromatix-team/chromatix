"""
Tests copied from [this file in MacroMax](https://github.com/tttom/MacroMax/blob/master/python/tests/test_hsv.py) and
adapted for JAX.
"""
import unittest

import numpy.testing as npt
import jax.numpy as jnp

from chromatix.utils.display import hsv2rgb, rgb2hsv


class TestHsv2Rgb(unittest.TestCase):
    def test_scalar(self):
        npt.assert_array_equal(hsv2rgb(0, 0, 0), [0, 0, 0])
        npt.assert_array_equal(hsv2rgb(0, 1, 0), [0, 0, 0])
        npt.assert_array_equal(hsv2rgb(1, 0, 0), [0, 0, 0])
        npt.assert_array_equal(hsv2rgb(0, 0, 1), [1, 1, 1])
        npt.assert_array_equal(hsv2rgb(0, 0, 0.5), [0.5, 0.5, 0.5])
        npt.assert_array_equal(hsv2rgb(0, 0, 1.5), [1.5, 1.5, 1.5])
        npt.assert_array_equal(hsv2rgb(0, 1, 1), [1, 0, 0])
        npt.assert_array_equal(hsv2rgb(0, 0.5, 1), [1, 0.5, 0.5])
        npt.assert_array_equal(hsv2rgb(0, 1, 1), [1, 0, 0])
        npt.assert_array_equal(hsv2rgb(1, 1, 1), [1, 0, 0])
        npt.assert_array_equal(hsv2rgb(1/6, 1, 1), [1, 1, 0])
        npt.assert_array_equal(hsv2rgb(2/6, 1, 1), [0, 1, 0])
        npt.assert_array_equal(hsv2rgb(3/6, 1, 1), [0, 1, 1])
        npt.assert_array_equal(hsv2rgb(4/6, 1, 1), [0, 0, 1])
        npt.assert_array_equal(hsv2rgb(5/6, 1, 1), [1, 0, 1])

    def test_vector(self):
        npt.assert_array_equal(hsv2rgb(jnp.ones(4), jnp.ones(4), jnp.ones(4)),
                               jnp.concatenate((jnp.ones((4, 1)), jnp.zeros((4, 1)), jnp.zeros((4, 1))), axis=-1))

        npt.assert_array_equal(hsv2rgb([0], [0], [0]), [[0, 0, 0]])
        npt.assert_array_equal(hsv2rgb([0], [0], [1]), [[1, 1, 1]])
        npt.assert_array_equal(hsv2rgb([0], [0], [0.5]), [[0.5, 0.5, 0.5]])
        npt.assert_array_equal(hsv2rgb([0], [0], [1.5]), [[1.5, 1.5, 1.5]])
        npt.assert_array_equal(hsv2rgb([0], [1], [1]), [[1, 0, 0]])
        npt.assert_array_equal(hsv2rgb([0], [0.5], [1]), [[1, 0.5, 0.5]])
        npt.assert_array_equal(hsv2rgb([0], [1], [1]), [[1, 0, 0]])
        npt.assert_array_equal(hsv2rgb([1], [1], [1]), [[1, 0, 0]])
        npt.assert_array_equal(hsv2rgb([1/6], [1], [1]), [[1, 1, 0]])
        npt.assert_array_equal(hsv2rgb([2/6], [1], [1]), [[0, 1, 0]])
        npt.assert_array_equal(hsv2rgb([3/6], [1], [1]), [[0, 1, 1]])
        npt.assert_array_equal(hsv2rgb([4/6], [1], [1]), [[0, 0, 1]])
        npt.assert_array_equal(hsv2rgb([5/6], [1], [1]), [[1, 0, 1]])

        npt.assert_array_equal(hsv2rgb([0], [0], [0]), [[0, 0, 0]])

    def test_matrix(self):
        npt.assert_array_equal(hsv2rgb(jnp.ones((5, 4)), jnp.ones((5, 4)), jnp.ones((5, 4))),
                               jnp.concatenate((jnp.ones((5, 4, 1)), jnp.zeros((5, 4, 1)), jnp.zeros((5, 4, 1))), axis=-1))

        npt.assert_array_equal(hsv2rgb([[0]], [[0]], [[0]]), [[[0, 0, 0]]])
        npt.assert_array_equal(hsv2rgb([[0]], [[0]], [[1]]), [[[1, 1, 1]]])
        npt.assert_array_equal(hsv2rgb([[0]], [[0]], [[0.5]]), [[[0.5, 0.5, 0.5]]])
        npt.assert_array_equal(hsv2rgb([[0]], [[0]], [[1.5]]), [[[1.5, 1.5, 1.5]]])
        npt.assert_array_equal(hsv2rgb([[0]], [[1]], [[1]]), [[[1, 0, 0]]])
        npt.assert_array_equal(hsv2rgb([[0]], [[0.5]], [[1]]), [[[1, 0.5, 0.5]]])
        npt.assert_array_equal(hsv2rgb([[0]], [[1]], [[1]]), [[[1, 0, 0]]])
        npt.assert_array_equal(hsv2rgb([[1]], [[1]], [[1]]), [[[1, 0, 0]]])
        npt.assert_array_equal(hsv2rgb([[1/6]], [[1]], [[1]]), [[[1, 1, 0]]])
        npt.assert_array_equal(hsv2rgb([[2/6]], [[1]], [[1]]), [[[0, 1, 0]]])
        npt.assert_array_equal(hsv2rgb([[3/6]], [[1]], [[1]]), [[[0, 1, 1]]])
        npt.assert_array_equal(hsv2rgb([[4/6]], [[1]], [[1]]), [[[0, 0, 1]]])
        npt.assert_array_equal(hsv2rgb([[5/6]], [[1]], [[1]]), [[[1, 0, 1]]])

        npt.assert_array_equal(hsv2rgb([[0]], [[0]], [[0]]), [[[0, 0, 0]]])

    def test_tensor(self):
        npt.assert_array_equal(hsv2rgb(jnp.ones((6, 5, 4)), jnp.ones((6, 5, 4)), jnp.ones((6, 5, 4))),
                               jnp.concatenate((jnp.ones((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1))), axis=-1))
        npt.assert_array_equal(hsv2rgb(jnp.ones((6, 5, 4)), 1, jnp.ones((6, 5, 4))),
                       jnp.concatenate((jnp.ones((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1))), axis=-1))
        npt.assert_array_equal(hsv2rgb(jnp.ones((6, 5, 4)), jnp.ones((6, 5, 4)), 1),
                               jnp.concatenate((jnp.ones((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1))), axis=-1))
        npt.assert_array_equal(hsv2rgb(1, jnp.ones((6, 5, 4)), 1),
                               jnp.concatenate((jnp.ones((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1))), axis=-1))

        npt.assert_array_equal(hsv2rgb([[[0]]], [[[0]]], [[[0]]]), [[[[0, 0, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[0]]], [[[0]]], [[[1]]]), [[[[1, 1, 1]]]])
        npt.assert_array_equal(hsv2rgb([[[0]]], [[[0]]], [[[0.5]]]), [[[[0.5, 0.5, 0.5]]]])
        npt.assert_array_equal(hsv2rgb([[[0]]], [[[0]]], [[[1.5]]]), [[[[1.5, 1.5, 1.5]]]])
        npt.assert_array_equal(hsv2rgb([[[0]]], [[[1]]], [[[1]]]), [[[[1, 0, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[0]]], [[[0.5]]], [[[1]]]), [[[[1, 0.5, 0.5]]]])
        npt.assert_array_equal(hsv2rgb([[[0]]], [[[1]]], [[[1]]]), [[[[1, 0, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[1]]], [[[1]]], [[[1]]]), [[[[1, 0, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[1/6]]], [[[1]]], [[[1]]]), [[[[1, 1, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[2/6]]], [[[1]]], [[[1]]]), [[[[0, 1, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[3/6]]], [[[1]]], [[[1]]]), [[[[0, 1, 1]]]])
        npt.assert_array_equal(hsv2rgb([[[4/6]]], [[[1]]], [[[1]]]), [[[[0, 0, 1]]]])
        npt.assert_array_equal(hsv2rgb([[[5/6]]], [[[1]]], [[[1]]]), [[[[1, 0, 1]]]])

        npt.assert_array_equal(hsv2rgb([[[0]]], [[[0]]], [[[0]]]), [[[[0, 0, 0]]]])


class TestRgb2Hsv(unittest.TestCase):
    def test_scalar(self):
        npt.assert_array_equal(rgb2hsv(0, 0, 0), [0, 0, 0])
        npt.assert_array_equal(rgb2hsv(1, 1, 1), [0, 0, 1])
        npt.assert_array_equal(rgb2hsv(0.5, 0.5, 0.5), [0, 0, 0.5])
        npt.assert_array_equal(rgb2hsv(1.5, 1.5, 1.5), [0, 0, 1.5])
        npt.assert_array_equal(rgb2hsv(1, 0, 0), [0, 1, 1])
        npt.assert_array_equal(rgb2hsv(1, 0.5, 0.5), [0, 0.5, 1])
        npt.assert_array_equal(rgb2hsv(1, 0, 0), [0, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv(1, 1, 0), [1/6, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv(0, 1, 0), [2/6, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv(0, 1, 1), [3/6, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv(0, 0, 1), [4/6, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv(1, 0, 1), [5/6, 1, 1])

    def test_vector(self):
        npt.assert_array_equal(rgb2hsv(jnp.ones(4), jnp.zeros(4), jnp.zeros(4)),
                               jnp.concatenate((jnp.zeros((4, 1)), jnp.ones((4, 1)), jnp.ones((4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([0], [0], [0]), [[0, 0, 0]])
        npt.assert_array_equal(rgb2hsv([1], [1], [1]), [[0, 0, 1]])
        npt.assert_array_equal(rgb2hsv([0.5], [0.5], [0.5]), [[0, 0, 0.5]])
        npt.assert_array_equal(rgb2hsv([1.5], [1.5], [1.5]), [[0, 0, 1.5]])
        npt.assert_array_equal(rgb2hsv([1], [0], [0]), [[0, 1, 1]])
        npt.assert_array_equal(rgb2hsv([1], [0.5], [0.5]), [[0, 0.5, 1]])
        npt.assert_array_equal(rgb2hsv([1], [0], [0]), [[0, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([1], [1], [0]), [[1/6, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([0], [1], [0]), [[2/6, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([0], [1], [1]), [[3/6, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([0], [0], [1]), [[4/6, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([1], [0], [1]), [[5/6, 1, 1]])

    def test_matrix(self):
        npt.assert_array_equal(rgb2hsv(jnp.ones((5, 4)), jnp.zeros((5, 4)), jnp.zeros((5, 4))),
                               jnp.concatenate((jnp.zeros((5, 4, 1)), jnp.ones((5, 4, 1)), jnp.ones((5, 4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[0]], [[0]], [[0]]), [[[0, 0, 0]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[1]], [[1]]), [[[0, 0, 1]]])
        npt.assert_array_equal(rgb2hsv([[0.5]], [[0.5]], [[0.5]]), [[[0, 0, 0.5]]])
        npt.assert_array_equal(rgb2hsv([[1.5]], [[1.5]], [[1.5]]), [[[0, 0, 1.5]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[0]], [[0]]), [[[0, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[0.5]], [[0.5]]), [[[0, 0.5, 1]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[0]], [[0]]), [[[0, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[1]], [[1]], [[0]]), [[[1/6, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[0]], [[1]], [[0]]), [[[2/6, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[0]], [[1]], [[1]]), [[[3/6, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[0]], [[0]], [[1]]), [[[4/6, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[1]], [[0]], [[1]]), [[[5/6, 1, 1]]])

    def test_tensor(self):
        npt.assert_array_equal(
            rgb2hsv(jnp.ones((6, 5, 4)), jnp.zeros((6, 5, 4)), jnp.zeros((6, 5, 4))),
                    jnp.concatenate((jnp.zeros((6, 5, 4, 1)), jnp.ones((6, 5, 4, 1)), jnp.ones((6, 5, 4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[[0]]], [[[0]]], [[[0]]]), [[[[0, 0, 0]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[1]]], [[[1]]]), [[[[0, 0, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[0.5]]], [[[0.5]]], [[[0.5]]]), [[[[0, 0, 0.5]]]])
        npt.assert_array_equal(rgb2hsv([[[1.5]]], [[[1.5]]], [[[1.5]]]), [[[[0, 0, 1.5]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[0]]], [[[0]]]), [[[[0, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[0.5]]], [[[0.5]]]), [[[[0, 0.5, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[0]]], [[[0]]]), [[[[0, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[1]]], [[[1]]], [[[0]]]), [[[[1/6, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[0]]], [[[1]]], [[[0]]]), [[[[2/6, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[0]]], [[[1]]], [[[1]]]), [[[[3/6, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[0]]], [[[0]]], [[[1]]]), [[[[4/6, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[1]]], [[[0]]], [[[1]]]), [[[[5/6, 1, 1]]]])


class TestHsv2RgbSingleArgument(unittest.TestCase):
    def test_scalar(self):
        npt.assert_array_equal(hsv2rgb([0, 0, 0]), [0, 0, 0])
        npt.assert_array_equal(hsv2rgb([0, 1, 0]), [0, 0, 0])
        npt.assert_array_equal(hsv2rgb([1, 0, 0]), [0, 0, 0])
        npt.assert_array_equal(hsv2rgb([0, 0, 1]), [1, 1, 1])
        npt.assert_array_equal(hsv2rgb([0, 0, 0.5]), [0.5, 0.5, 0.5])
        npt.assert_array_equal(hsv2rgb([0, 0, 1.5]), [1.5, 1.5, 1.5])
        npt.assert_array_equal(hsv2rgb([0, 1, 1]), [1, 0, 0])
        npt.assert_array_equal(hsv2rgb([0, 0.5, 1]), [1, 0.5, 0.5])
        npt.assert_array_equal(hsv2rgb([0, 1, 1]), [1, 0, 0])
        npt.assert_array_equal(hsv2rgb([1, 1, 1]), [1, 0, 0])
        npt.assert_array_equal(hsv2rgb([1/6, 1, 1]), [1, 1, 0])
        npt.assert_array_equal(hsv2rgb([2/6, 1, 1]), [0, 1, 0])
        npt.assert_array_equal(hsv2rgb([3/6, 1, 1]), [0, 1, 1])
        npt.assert_array_equal(hsv2rgb([4/6, 1, 1]), [0, 0, 1])
        npt.assert_array_equal(hsv2rgb([5/6, 1, 1]), [1, 0, 1])

    def test_vector(self):
        npt.assert_array_equal(hsv2rgb(jnp.ones((4, 3))),
                               jnp.concatenate((jnp.ones((4, 1)), jnp.zeros((4, 1)), jnp.zeros((4, 1))), axis=-1))

        npt.assert_array_equal(hsv2rgb([[0, 0, 0]]), [[0, 0, 0]])
        npt.assert_array_equal(hsv2rgb([[0, 0, 1]]), [[1, 1, 1]])
        npt.assert_array_equal(hsv2rgb([[0, 0, 0.5]]), [[0.5, 0.5, 0.5]])
        npt.assert_array_equal(hsv2rgb([[0, 0, 1.5]]), [[1.5, 1.5, 1.5]])
        npt.assert_array_equal(hsv2rgb([[0, 1, 1]]), [[1, 0, 0]])
        npt.assert_array_equal(hsv2rgb([[0, 0.5, 1]]), [[1, 0.5, 0.5]])
        npt.assert_array_equal(hsv2rgb([[0, 1, 1]]), [[1, 0, 0]])
        npt.assert_array_equal(hsv2rgb([[1, 1, 1]]), [[1, 0, 0]])
        npt.assert_array_equal(hsv2rgb([[1/6, 1, 1]]), [[1, 1, 0]])
        npt.assert_array_equal(hsv2rgb([[2/6, 1, 1]]), [[0, 1, 0]])
        npt.assert_array_equal(hsv2rgb([[3/6, 1, 1]]), [[0, 1, 1]])
        npt.assert_array_equal(hsv2rgb([[4/6, 1, 1]]), [[0, 0, 1]])
        npt.assert_array_equal(hsv2rgb([[5/6, 1, 1]]), [[1, 0, 1]])

        npt.assert_array_equal(hsv2rgb([[0, 0, 0]]), [[0, 0, 0]])

    def test_matrix(self):
        npt.assert_array_equal(hsv2rgb(jnp.ones((5, 4, 3))),
                               jnp.concatenate((jnp.ones((5, 4, 1)), jnp.zeros((5, 4, 1)), jnp.zeros((5, 4, 1))), axis=-1))

        npt.assert_array_equal(hsv2rgb([[[0, 0, 0]]]), [[[0, 0, 0]]])
        npt.assert_array_equal(hsv2rgb([[[0, 0, 1]]]), [[[1, 1, 1]]])
        npt.assert_array_equal(hsv2rgb([[[0, 0, 0.5]]]), [[[0.5, 0.5, 0.5]]])
        npt.assert_array_equal(hsv2rgb([[[0, 0, 1.5]]]), [[[1.5, 1.5, 1.5]]])
        npt.assert_array_equal(hsv2rgb([[[0, 1, 1]]]), [[[1, 0, 0]]])
        npt.assert_array_equal(hsv2rgb([[[0, 0.5, 1]]]), [[[1, 0.5, 0.5]]])
        npt.assert_array_equal(hsv2rgb([[[0, 1, 1]]]), [[[1, 0, 0]]])
        npt.assert_array_equal(hsv2rgb([[[1, 1, 1]]]), [[[1, 0, 0]]])
        npt.assert_array_equal(hsv2rgb([[[1/6, 1, 1]]]), [[[1, 1, 0]]])
        npt.assert_array_equal(hsv2rgb([[[2/6, 1, 1]]]), [[[0, 1, 0]]])
        npt.assert_array_equal(hsv2rgb([[[3/6, 1, 1]]]), [[[0, 1, 1]]])
        npt.assert_array_equal(hsv2rgb([[[4/6, 1, 1]]]), [[[0, 0, 1]]])
        npt.assert_array_equal(hsv2rgb([[[5/6, 1, 1]]]), [[[1, 0, 1]]])

        npt.assert_array_equal(hsv2rgb([[0, 0, 0]]), [[0, 0, 0]])

    def test_tensor(self):
        npt.assert_array_equal(hsv2rgb(jnp.ones((6, 5, 4, 3))),
                               jnp.concatenate((jnp.ones((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1))), axis=-1))

        npt.assert_array_equal(hsv2rgb([[[[0, 0, 0]]]]), [[[[0, 0, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[[0, 0, 1]]]]), [[[[1, 1, 1]]]])
        npt.assert_array_equal(hsv2rgb([[[[0, 0, 0.5]]]]), [[[[0.5, 0.5, 0.5]]]])
        npt.assert_array_equal(hsv2rgb([[[[0, 0, 1.5]]]]), [[[[1.5, 1.5, 1.5]]]])
        npt.assert_array_equal(hsv2rgb([[[[0, 1, 1]]]]), [[[[1, 0, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[[0, 0.5, 1]]]]), [[[[1, 0.5, 0.5]]]])
        npt.assert_array_equal(hsv2rgb([[[[0, 1, 1]]]]), [[[[1, 0, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[[1, 1, 1]]]]), [[[[1, 0, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[[1/6, 1, 1]]]]), [[[[1, 1, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[[2/6, 1, 1]]]]), [[[[0, 1, 0]]]])
        npt.assert_array_equal(hsv2rgb([[[[3/6, 1, 1]]]]), [[[[0, 1, 1]]]])
        npt.assert_array_equal(hsv2rgb([[[[4/6, 1, 1]]]]), [[[[0, 0, 1]]]])
        npt.assert_array_equal(hsv2rgb([[[[5/6, 1, 1]]]]), [[[[1, 0, 1]]]])

        npt.assert_array_equal(hsv2rgb([[[[0, 0, 0]]]]), [[[[0, 0, 0]]]])


class TestRgb2HsvSingleArgument(unittest.TestCase):
    def test_scalar(self):
        npt.assert_array_equal(rgb2hsv([0, 0, 0]), [0, 0, 0])
        npt.assert_array_equal(rgb2hsv([1, 1, 1]), [0, 0, 1])
        npt.assert_array_equal(rgb2hsv([0.5, 0.5, 0.5]), [0, 0, 0.5])
        npt.assert_array_equal(rgb2hsv([1.5, 1.5, 1.5]), [0, 0, 1.5])
        npt.assert_array_equal(rgb2hsv([1, 0, 0]), [0, 1, 1])
        npt.assert_array_equal(rgb2hsv([1, 0.5, 0.5]), [0, 0.5, 1])
        npt.assert_array_equal(rgb2hsv([1, 0, 0]), [0, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv([1, 1, 0]), [1/6, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv([0, 1, 0]), [2/6, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv([0, 1, 1]), [3/6, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv([0, 0, 1]), [4/6, 1, 1])
        npt.assert_array_almost_equal(rgb2hsv([1, 0, 1]), [5/6, 1, 1])

    def test_vector(self):
        npt.assert_array_equal(rgb2hsv(jnp.concatenate((jnp.ones((4, 1)), jnp.zeros((4, 1)), jnp.zeros((4, 1))), axis=-1)),
                               jnp.concatenate((jnp.zeros((4, 1)), jnp.ones((4, 1)), jnp.ones((4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[0, 0, 0]]), [[0, 0, 0]])
        npt.assert_array_equal(rgb2hsv([[1, 1, 1]]), [[0, 0, 1]])
        npt.assert_array_equal(rgb2hsv([[0.5, 0.5, 0.5]]), [[0, 0, 0.5]])
        npt.assert_array_equal(rgb2hsv([[1.5, 1.5, 1.5]]), [[0, 0, 1.5]])
        npt.assert_array_equal(rgb2hsv([[1, 0, 0]]), [[0, 1, 1]])
        npt.assert_array_equal(rgb2hsv([[1, 0.5, 0.5]]), [[0, 0.5, 1]])
        npt.assert_array_equal(rgb2hsv([[1, 0, 0]]), [[0, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([[1, 1, 0]]), [[1/6, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([[0, 1, 0]]), [[2/6, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([[0, 1, 1]]), [[3/6, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([[0, 0, 1]]), [[4/6, 1, 1]])
        npt.assert_array_almost_equal(rgb2hsv([[1, 0, 1]]), [[5/6, 1, 1]])

    def test_matrix(self):
        npt.assert_array_equal(rgb2hsv(jnp.concatenate((jnp.ones((5, 4, 1)), jnp.zeros((5, 4, 1)), jnp.zeros((5, 4, 1))), axis=-1)),
                               jnp.concatenate((jnp.zeros((5, 4, 1)), jnp.ones((5, 4, 1)), jnp.ones((5, 4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[[0, 0, 0]]]), [[[0, 0, 0]]])
        npt.assert_array_equal(rgb2hsv([[[1, 1, 1]]]), [[[0, 0, 1]]])
        npt.assert_array_equal(rgb2hsv([[[0.5, 0.5, 0.5]]]), [[[0, 0, 0.5]]])
        npt.assert_array_equal(rgb2hsv([[[1.5, 1.5, 1.5]]]), [[[0, 0, 1.5]]])
        npt.assert_array_equal(rgb2hsv([[[1, 0, 0]]]), [[[0, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[[1, 0.5, 0.5]]]), [[[0, 0.5, 1]]])
        npt.assert_array_equal(rgb2hsv([[[1, 0, 0]]]), [[[0, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[[1, 1, 0]]]), [[[1/6, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[[0, 1, 0]]]), [[[2/6, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[[0, 1, 1]]]), [[[3/6, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[[0, 0, 1]]]), [[[4/6, 1, 1]]])
        npt.assert_array_almost_equal(rgb2hsv([[[1, 0, 1]]]), [[[5/6, 1, 1]]])

    def test_tensor(self):
        npt.assert_array_equal(
            rgb2hsv(jnp.concatenate((jnp.ones((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1)), jnp.zeros((6, 5, 4, 1))), axis=-1)),
            jnp.concatenate((jnp.zeros((6, 5, 4, 1)), jnp.ones((6, 5, 4, 1)), jnp.ones((6, 5, 4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[[[0, 0, 0]]]]), [[[[0, 0, 0]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 1, 1]]]]), [[[[0, 0, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[0.5, 0.5, 0.5]]]]), [[[[0, 0, 0.5]]]])
        npt.assert_array_equal(rgb2hsv([[[[1.5, 1.5, 1.5]]]]), [[[[0, 0, 1.5]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 0, 0]]]]), [[[[0, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 0.5, 0.5]]]]), [[[[0, 0.5, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 0, 0]]]]), [[[[0, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[[1, 1, 0]]]]), [[[[1/6, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[[0, 1, 0]]]]), [[[[2/6, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[[0, 1, 1]]]]), [[[[3/6, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[[0, 0, 1]]]]), [[[[4/6, 1, 1]]]])
        npt.assert_array_almost_equal(rgb2hsv([[[[1, 0, 1]]]]), [[[[5/6, 1, 1]]]])


if __name__ == '__main__':
    unittest.main()
