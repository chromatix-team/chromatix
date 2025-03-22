import unittest
import numpy.testing as npt
import numpy as np

from macromax.utils.display import hsv2rgb, rgb2hsv


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
        npt.assert_array_equal(hsv2rgb(np.ones(4), np.ones(4), np.ones(4)),
                               np.concatenate((np.ones((4, 1)), np.zeros((4, 1)), np.zeros((4, 1))), axis=-1))

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
        npt.assert_array_equal(hsv2rgb(np.ones((5, 4)), np.ones((5, 4)), np.ones((5, 4))),
                               np.concatenate((np.ones((5, 4, 1)), np.zeros((5, 4, 1)), np.zeros((5, 4, 1))), axis=-1))

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
        npt.assert_array_equal(hsv2rgb(np.ones((6, 5, 4)), np.ones((6, 5, 4)), np.ones((6, 5, 4))),
                               np.concatenate((np.ones((6, 5, 4, 1)), np.zeros((6, 5, 4, 1)), np.zeros((6, 5, 4, 1))), axis=-1))
        npt.assert_array_equal(hsv2rgb(np.ones((6, 5, 4)), 1, np.ones((6, 5, 4))),
                       np.concatenate((np.ones((6, 5, 4, 1)), np.zeros((6, 5, 4, 1)), np.zeros((6, 5, 4, 1))), axis=-1))
        npt.assert_array_equal(hsv2rgb(np.ones((6, 5, 4)), np.ones((6, 5, 4)), 1),
                               np.concatenate((np.ones((6, 5, 4, 1)), np.zeros((6, 5, 4, 1)), np.zeros((6, 5, 4, 1))), axis=-1))
        npt.assert_array_equal(hsv2rgb(1, np.ones((6, 5, 4)), 1),
                               np.concatenate((np.ones((6, 5, 4, 1)), np.zeros((6, 5, 4, 1)), np.zeros((6, 5, 4, 1))), axis=-1))

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
        npt.assert_array_equal(rgb2hsv(1, 1, 0), [1/6, 1, 1])
        npt.assert_array_equal(rgb2hsv(0, 1, 0), [2/6, 1, 1])
        npt.assert_array_equal(rgb2hsv(0, 1, 1), [3/6, 1, 1])
        npt.assert_array_equal(rgb2hsv(0, 0, 1), [4/6, 1, 1])
        npt.assert_array_equal(rgb2hsv(1, 0, 1), [5/6, 1, 1])

    def test_vector(self):
        npt.assert_array_equal(rgb2hsv(np.ones(4), np.zeros(4), np.zeros(4)),
                               np.concatenate((np.zeros((4, 1)), np.ones((4, 1)), np.ones((4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([0], [0], [0]), [[0, 0, 0]])
        npt.assert_array_equal(rgb2hsv([1], [1], [1]), [[0, 0, 1]])
        npt.assert_array_equal(rgb2hsv([0.5], [0.5], [0.5]), [[0, 0, 0.5]])
        npt.assert_array_equal(rgb2hsv([1.5], [1.5], [1.5]), [[0, 0, 1.5]])
        npt.assert_array_equal(rgb2hsv([1], [0], [0]), [[0, 1, 1]])
        npt.assert_array_equal(rgb2hsv([1], [0.5], [0.5]), [[0, 0.5, 1]])
        npt.assert_array_equal(rgb2hsv([1], [0], [0]), [[0, 1, 1]])
        npt.assert_array_equal(rgb2hsv([1], [1], [0]), [[1/6, 1, 1]])
        npt.assert_array_equal(rgb2hsv([0], [1], [0]), [[2/6, 1, 1]])
        npt.assert_array_equal(rgb2hsv([0], [1], [1]), [[3/6, 1, 1]])
        npt.assert_array_equal(rgb2hsv([0], [0], [1]), [[4/6, 1, 1]])
        npt.assert_array_equal(rgb2hsv([1], [0], [1]), [[5/6, 1, 1]])

    def test_matrix(self):
        npt.assert_array_equal(rgb2hsv(np.ones((5, 4)), np.zeros((5, 4)), np.zeros((5, 4))),
                               np.concatenate((np.zeros((5, 4, 1)), np.ones((5, 4, 1)), np.ones((5, 4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[0]], [[0]], [[0]]), [[[0, 0, 0]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[1]], [[1]]), [[[0, 0, 1]]])
        npt.assert_array_equal(rgb2hsv([[0.5]], [[0.5]], [[0.5]]), [[[0, 0, 0.5]]])
        npt.assert_array_equal(rgb2hsv([[1.5]], [[1.5]], [[1.5]]), [[[0, 0, 1.5]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[0]], [[0]]), [[[0, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[0.5]], [[0.5]]), [[[0, 0.5, 1]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[0]], [[0]]), [[[0, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[1]], [[0]]), [[[1/6, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[0]], [[1]], [[0]]), [[[2/6, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[0]], [[1]], [[1]]), [[[3/6, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[0]], [[0]], [[1]]), [[[4/6, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[1]], [[0]], [[1]]), [[[5/6, 1, 1]]])

    def test_tensor(self):
        npt.assert_array_equal(
            rgb2hsv(np.ones((6, 5, 4)), np.zeros((6, 5, 4)), np.zeros((6, 5, 4))),
                    np.concatenate((np.zeros((6, 5, 4, 1)), np.ones((6, 5, 4, 1)), np.ones((6, 5, 4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[[0]]], [[[0]]], [[[0]]]), [[[[0, 0, 0]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[1]]], [[[1]]]), [[[[0, 0, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[0.5]]], [[[0.5]]], [[[0.5]]]), [[[[0, 0, 0.5]]]])
        npt.assert_array_equal(rgb2hsv([[[1.5]]], [[[1.5]]], [[[1.5]]]), [[[[0, 0, 1.5]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[0]]], [[[0]]]), [[[[0, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[0.5]]], [[[0.5]]]), [[[[0, 0.5, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[0]]], [[[0]]]), [[[[0, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[1]]], [[[0]]]), [[[[1/6, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[0]]], [[[1]]], [[[0]]]), [[[[2/6, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[0]]], [[[1]]], [[[1]]]), [[[[3/6, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[0]]], [[[0]]], [[[1]]]), [[[[4/6, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[1]]], [[[0]]], [[[1]]]), [[[[5/6, 1, 1]]]])


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
        npt.assert_array_equal(hsv2rgb(np.ones((4, 3))),
                               np.concatenate((np.ones((4, 1)), np.zeros((4, 1)), np.zeros((4, 1))), axis=-1))

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
        npt.assert_array_equal(hsv2rgb(np.ones((5, 4, 3))),
                               np.concatenate((np.ones((5, 4, 1)), np.zeros((5, 4, 1)), np.zeros((5, 4, 1))), axis=-1))

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
        npt.assert_array_equal(hsv2rgb(np.ones((6, 5, 4, 3))),
                               np.concatenate((np.ones((6, 5, 4, 1)), np.zeros((6, 5, 4, 1)), np.zeros((6, 5, 4, 1))), axis=-1))

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
        npt.assert_array_equal(rgb2hsv([1, 1, 0]), [1/6, 1, 1])
        npt.assert_array_equal(rgb2hsv([0, 1, 0]), [2/6, 1, 1])
        npt.assert_array_equal(rgb2hsv([0, 1, 1]), [3/6, 1, 1])
        npt.assert_array_equal(rgb2hsv([0, 0, 1]), [4/6, 1, 1])
        npt.assert_array_equal(rgb2hsv([1, 0, 1]), [5/6, 1, 1])

    def test_vector(self):
        npt.assert_array_equal(rgb2hsv(np.concatenate((np.ones((4, 1)), np.zeros((4, 1)), np.zeros((4, 1))), axis=-1)),
                               np.concatenate((np.zeros((4, 1)), np.ones((4, 1)), np.ones((4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[0, 0, 0]]), [[0, 0, 0]])
        npt.assert_array_equal(rgb2hsv([[1, 1, 1]]), [[0, 0, 1]])
        npt.assert_array_equal(rgb2hsv([[0.5, 0.5, 0.5]]), [[0, 0, 0.5]])
        npt.assert_array_equal(rgb2hsv([[1.5, 1.5, 1.5]]), [[0, 0, 1.5]])
        npt.assert_array_equal(rgb2hsv([[1, 0, 0]]), [[0, 1, 1]])
        npt.assert_array_equal(rgb2hsv([[1, 0.5, 0.5]]), [[0, 0.5, 1]])
        npt.assert_array_equal(rgb2hsv([[1, 0, 0]]), [[0, 1, 1]])
        npt.assert_array_equal(rgb2hsv([[1, 1, 0]]), [[1/6, 1, 1]])
        npt.assert_array_equal(rgb2hsv([[0, 1, 0]]), [[2/6, 1, 1]])
        npt.assert_array_equal(rgb2hsv([[0, 1, 1]]), [[3/6, 1, 1]])
        npt.assert_array_equal(rgb2hsv([[0, 0, 1]]), [[4/6, 1, 1]])
        npt.assert_array_equal(rgb2hsv([[1, 0, 1]]), [[5/6, 1, 1]])

    def test_matrix(self):
        npt.assert_array_equal(rgb2hsv(np.concatenate((np.ones((5, 4, 1)), np.zeros((5, 4, 1)), np.zeros((5, 4, 1))), axis=-1)),
                               np.concatenate((np.zeros((5, 4, 1)), np.ones((5, 4, 1)), np.ones((5, 4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[[0, 0, 0]]]), [[[0, 0, 0]]])
        npt.assert_array_equal(rgb2hsv([[[1, 1, 1]]]), [[[0, 0, 1]]])
        npt.assert_array_equal(rgb2hsv([[[0.5, 0.5, 0.5]]]), [[[0, 0, 0.5]]])
        npt.assert_array_equal(rgb2hsv([[[1.5, 1.5, 1.5]]]), [[[0, 0, 1.5]]])
        npt.assert_array_equal(rgb2hsv([[[1, 0, 0]]]), [[[0, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[[1, 0.5, 0.5]]]), [[[0, 0.5, 1]]])
        npt.assert_array_equal(rgb2hsv([[[1, 0, 0]]]), [[[0, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[[1, 1, 0]]]), [[[1/6, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[[0, 1, 0]]]), [[[2/6, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[[0, 1, 1]]]), [[[3/6, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[[0, 0, 1]]]), [[[4/6, 1, 1]]])
        npt.assert_array_equal(rgb2hsv([[[1, 0, 1]]]), [[[5/6, 1, 1]]])

    def test_tensor(self):
        npt.assert_array_equal(
            rgb2hsv(np.concatenate((np.ones((6, 5, 4, 1)), np.zeros((6, 5, 4, 1)), np.zeros((6, 5, 4, 1))), axis=-1)),
            np.concatenate((np.zeros((6, 5, 4, 1)), np.ones((6, 5, 4, 1)), np.ones((6, 5, 4, 1))), axis=-1))

        npt.assert_array_equal(rgb2hsv([[[[0, 0, 0]]]]), [[[[0, 0, 0]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 1, 1]]]]), [[[[0, 0, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[0.5, 0.5, 0.5]]]]), [[[[0, 0, 0.5]]]])
        npt.assert_array_equal(rgb2hsv([[[[1.5, 1.5, 1.5]]]]), [[[[0, 0, 1.5]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 0, 0]]]]), [[[[0, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 0.5, 0.5]]]]), [[[[0, 0.5, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 0, 0]]]]), [[[[0, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 1, 0]]]]), [[[[1/6, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[0, 1, 0]]]]), [[[[2/6, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[0, 1, 1]]]]), [[[[3/6, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[0, 0, 1]]]]), [[[[4/6, 1, 1]]]])
        npt.assert_array_equal(rgb2hsv([[[[1, 0, 1]]]]), [[[[5/6, 1, 1]]]])


if __name__ == '__main__':
    unittest.main()
