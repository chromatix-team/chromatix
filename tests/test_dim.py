"""
Tests copied from [this file in MacroMax](https://github.com/tttom/MacroMax/blob/master/python/tests/test_utils.py) and
adapted for JAX.
"""
import unittest

import jax.numpy as jnp
import numpy.testing as npt

from chromatix.utils import dim


class TestDim(unittest.TestCase):
    def test_vector_to_axis(self):
        a = dim.to_axis(jnp.asarray([3, 1, 4]), axis=0, ndim=1)
        npt.assert_array_almost_equal(a, jnp.asarray([3, 1, 4]))

        a = dim.to_axis(jnp.asarray([3, 1, 4]), ndim=2)
        npt.assert_array_almost_equal(a, jnp.asarray([[3], [1], [4]]))

        a = dim.to_axis(jnp.asarray([3, 1, 4]), axis=1, ndim=2)
        npt.assert_array_almost_equal(a, jnp.asarray([[3, 1, 4]]))

        a = dim.to_axis(jnp.asarray(3), axis=0, ndim=1)
        npt.assert_almost_equal(a, jnp.asarray([3]))

    def test_add_dims_on_right(self):
        npt.assert_array_equal(dim.add([1, 2, 3], right=1), jnp.asarray([1, 2, 3])[..., jnp.newaxis], err_msg='Could not add 1 axis to a vector.')
        npt.assert_array_equal(dim.add([1, 2, 3], right=2), jnp.asarray([1, 2, 3])[..., jnp.newaxis, jnp.newaxis], err_msg='Could not add 3 axes to a vector.')
        npt.assert_array_equal(dim.add([1, 2, 3], right=3), jnp.asarray([1, 2, 3])[..., jnp.newaxis, jnp.newaxis, jnp.newaxis], err_msg='Could not add 3 axes to a vector.')
        npt.assert_array_equal(dim.add([1, 2, 3], right=0), jnp.asarray([1, 2, 3]), err_msg='Could not add 0 axes')
        npt.assert_array_equal(dim.add([1, 2, 3]), jnp.asarray([1, 2, 3]), err_msg='Number of axes to add does not default to 0.')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]], right=1), jnp.asarray([[1, 2, 3], [4, 5, 6]])[..., jnp.newaxis], err_msg='Could not add 1 axis to a matrix.')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]], right=2), jnp.asarray([[1, 2, 3], [4, 5, 6]])[..., jnp.newaxis, jnp.newaxis], err_msg='Could not add 2 axes to a matrix.')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]], right=3), jnp.asarray([[1, 2, 3], [4, 5, 6]])[..., jnp.newaxis, jnp.newaxis, jnp.newaxis], err_msg='Could not add 3 axes to a matrix.')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]], right=0), jnp.asarray([[1, 2, 3], [4, 5, 6]]), err_msg='Could not add 0 axes')
        npt.assert_array_equal(dim.add([[1, 2, 3], [4, 5, 6]]), jnp.asarray([[1, 2, 3], [4, 5, 6]]), err_msg='Number of axes to add does not default to 0.')
        npt.assert_array_equal(dim.add([[1], [4]], right=-1), jnp.asarray([1, 4]), err_msg='Could not remove axes on the right')
        npt.assert_array_equal(dim.add([1], right=1), jnp.asarray([1])[..., jnp.newaxis], err_msg='Could not add axis to [1].')
        npt.assert_array_equal(dim.add([[1]], right=1), jnp.asarray([[1]])[..., jnp.newaxis], err_msg='Could not add axis to [[1]].')
        npt.assert_array_equal(dim.add(1, 1), jnp.asarray(1)[..., jnp.newaxis], err_msg='Could not add axis to a scalar.')
        npt.assert_array_equal(dim.add([], right=1), jnp.asarray([])[..., jnp.newaxis], err_msg='Could not add axis to an empty array of shape (0,) .')


if __name__ == '__main__':
    unittest.main()
