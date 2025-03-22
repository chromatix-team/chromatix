import numpy as np

from typing import Union, Sequence, Optional
from numbers import Complex
array_like = Union[Complex, Sequence, np.ndarray]


def add(arr: array_like, left: Optional[int] = None, right: Optional[int] = None, ndim: Optional[int] = None
        ) -> np.ndarray:
    """
    A function that returns a view with additional singleton dimensions on the left or right. This is useful when
    stacking arrays along a new dimension that is further out to the left, or to broadcast and expand on the left-hand
    side with an array of an arbitrary number of dimensions on the right-hand side.

    :param arr: The original array or sequence of numbers that can be converted to an array.
    :param left: (optional) The number of axes to add on the left-hand side. Default: 0. When negative, singleton
        dimensions are removed from the left. This will fail if those on the left are not singleton dimensions.
    :param right: (optional) The number of axes to add on the right-hand side. Default: 0. When negative, singleton
        dimensions are removed from the right. This will fail if those on the right are not singleton dimensions.
    :param ndim: (optional) The total disired number of axes after adding those specified by ``left`` or ``right``. The
        remaining axes are added or removed at the right when ``right`` is not specified, if not, on the left when
        ``left`` is not specified. This argument is ignored if both ``left`` and ``right`` are specified.

    :return: A view with ndim == arr.ndim + left + right dimensions, where any singleton dimensions are added or removed
        on the left or right.
    """
    arr = np.asarray(arr)
    if ndim is not None:  # Work out the left and right padding from the total
        if right is None:
            right = ndim - arr.ndim
            if left is not None:
                right -= left
        elif left is None:
            left = ndim - arr.ndim
            if right is not None:
                left -= right
    if left is not None:
        # Add on left as required
        if left > 0:
            arr = np.expand_dims(arr, tuple(range(left)))
        elif left < 0:
            arr = arr.reshape(arr.shape[-left:])
    if right is not None:
        # Add on right as required
        if right > 0:
            arr = np.expand_dims(arr, tuple(range(-right, 0)))
        elif right < 0:
            arr = arr.reshape(arr.shape[:-right])

    return arr


def right_to(arr: array_like, ndim: int) -> np.ndarray:
    """
    A function that returns a view with additional singleton dimensions on the right up to a fixed number of dimensions.
    This is useful to broadcast and expand on the left-hand side with an array of an arbitrary number of dimensions on
    the right-hand side.

    :param arr: The original array or sequence of numbers that can be converted to an array.
    :param ndim: The total number of axes of the returned view.

    :return: A view with ndim == arr.ndim + right dimensions, where any singleton dimensions are added or removed on the right.
    """
    return add(arr, ndim=ndim)


def left_to(arr: array_like, ndim: int) -> np.ndarray:
    """
    A function that returns a view with additional singleton dimensions on the left up to a fixed number of dimensions.
    This is useful when stacking arrays along a new dimension that is further out to the left.

    :param arr: The original array or sequence of numbers that can be converted to an array.
    :param ndim: (optional) The total number of axes of the returned view.

    :return: A view with ndim == arr.ndim + left dimensions, where any singleton dimensions are added or removed on the left.
    """
    return add(arr, ndim=ndim, right=0)


def to_axis(arr: array_like, axis: int = 0, ndim: Optional[int] = None) -> np.ndarray:
    """
    Returns a view in which the first axis of the original array is at the target index. Negative indices count back
    from the right-most dimension in the view to the last axis of the original array. The total number of dimensions of
    the view can optionally be specified. This is useful to orient vectors along a desired dimensions.

    :param arr: the input array
    :param axis: the target axis for the first axis of the original array in the target view (default: 0)
        When negative values are specified, the axes are counted from the right-hand side to the last axis of the
        original array.
    :param ndim: the number of desired dimensions. Default: max(axis, 1-axis) + arr.ndim

    :return: an n-dimensional array view with all-but-one singleton dimension
    """
    arr = np.asarray(arr)
    if ndim is None:
        ndim = arr.ndim + max(axis, -axis-1)
    return add(arr, left=axis % ndim, ndim=ndim)
