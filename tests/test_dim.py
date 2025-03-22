import unittest
import numpy as np
import numpy.testing as npt

from macromax.utils import dim
from macromax.utils.display import complex2rgb, grid2extent, hsv2rgb


class TestUtils(unittest.TestCase):
    def test_vector_to_axis(self):
        a = dim.to_axis(np.array([3, 1, 4]), axis=0, ndim=1)
        npt.assert_almost_equal(a, np.array([3, 1, 4]))

        a = dim.to_axis(np.array([3, 1, 4]), ndim=2)
        npt.assert_almost_equal(a, np.array([[3], [1], [4]]))

        a = dim.to_axis(np.array([3, 1, 4]), axis=1, ndim=2)
        npt.assert_almost_equal(a, np.array([[3, 1, 4]]))

        a = dim.to_axis(np.array(3), axis=0, ndim=1)
        npt.assert_almost_equal(a, np.array([3]))

    def test_add_dims_on_right(self):
        npt.assert_array_equal(dim.add([1, 2, 3], right=1), np.array([1, 2, 3])[..., np.newaxis], err_msg='Could not add 1 axis to a vector.')
        npt.assert_array_equal(dim.add([1, 2, 3], right=2), np.array([1, 2, 3])[..., np.newaxis, np.newaxis], err_msg='Could not add 3 axes to a vector.')
        npt.assert_array_equal(dim.add([1, 2, 3], right=3), np.array([1, 2, 3])[..., np.newaxis, np.newaxis, np.newaxis], err_msg='Could not add 3 axes to a vector.')
        npt.assert_array_equal(dim.add([1, 2, 3], right=0), np.array([1, 2, 3]), err_msg='Could not add 0 axes')
        npt.assert_array_equal(dim.add([1, 2, 3]), np.array([1, 2, 3]), err_msg='Number of axes to add does not default to 0.')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]], right=1), np.array([[1, 2, 3], [4, 5, 6]])[..., np.newaxis], err_msg='Could not add 1 axis to a matrix.')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]], right=2), np.array([[1, 2, 3], [4, 5, 6]])[..., np.newaxis, np.newaxis], err_msg='Could not add 2 axes to a matrix.')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]], right=3), np.array([[1, 2, 3], [4, 5, 6]])[..., np.newaxis, np.newaxis, np.newaxis], err_msg='Could not add 3 axes to a matrix.')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]], right=0), np.array([[1, 2, 3], [4, 5, 6]]), err_msg='Could not add 0 axes')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]), err_msg='Number of axes to add does not default to 0.')
        npt.assert_array_equal(dim.add([[1], [4]], right=-1), np.array([1, 4]), err_msg='Could not remove axes on the right')
        npt.assert_array_equal(dim.add([1], right=1), np.array([1])[..., np.newaxis], err_msg='Could not add axis to [1].')
        npt.assert_array_equal(dim.add([[1]], right=1), np.array([[1]])[..., np.newaxis], err_msg='Could not add axis to [[1]].')
        npt.assert_array_equal(dim.add(1, 1), np.array(1)[..., np.newaxis], err_msg='Could not add axis to a scalar.')
        npt.assert_array_equal(dim.add([], right=1), np.array([])[..., np.newaxis], err_msg='Could not add axis to an empty array of shape (0,) .')


if __name__ == '__main__':
    unittest.main()
