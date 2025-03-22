"""
Code ported from [this file in MacroMax](https://github.com/tttom/MacroMax/blob/master/python/macromax/utils/ft/grid.py)
to work with JAX. For simplicity, only multidimensional Grids have been kept, which required substantial changes to
`test_grid.py`. Removed several checks to get it to JIT. Removed the MutableGrid class.
"""
from __future__ import annotations

from typing import Union, Sequence

import jax
import jax.numpy as jnp
from jax import tree_util

from chromatix.utils import dim

__all__ = ['Grid']


@tree_util.register_pytree_node_class
class Grid(Sequence):
    """
    A class representing an immutable uniformly-spaced plaid Cartesian grid and its Fourier Transform.
    """
    def __init__(self, shape=None, step=None, *, extent=None, first=None, center=None, last=None, include_last=False,
                 ndim: int = None,
                 flat: Union[bool, Sequence, jax.Array] = False,
                 origin_at_center: Union[bool, Sequence, jax.Array] = True,
                 center_at_index: Union[bool, Sequence, jax.Array] = True):
        """
        Construct an immutable `Grid` object.

        Its values are defined by `shape`, `step`, `center`, and the boolean flags `include_last` and`center_at_index`.
        If not specified, the values for `shape`, `step`, and `center` center are deduced from the other values,
        including `first`, `last`, and `extent`. A larger even shape is assumed in case of ambiguity. If all else fails,
        first the step is assumed to be 1, then the center is assumed to be as close as possible to 0, and finally the
        shape is assumed to be 1.

        Specific invariants:

            - ```shape * step == extent == last + step - first``` if `include_last`
            - ```shape * step == extent == last - first``` if `not include_last`
            - ```center == first + shape // 2 * step``` if `center_at_index`
            - ```center == first + (shape - 1) / 2 * step``` if `not center_at_index`

        General invariants:

            - ```shape * step == extent == last + step * include_last - first```
            - ```center == first + (shape // 2 * center_at_index + (shape - 1) / 2 * (1 - center_at_index)) * step```

        :param shape: An integer vector array with the shape of the sampling grid.
        :param step: A vector array with the spacing of the sampling grid. This defaults to 1 if no two of first,
            center, or last are specified.
        :param extent: The extent of the sampling grid as shape * step = last - first.
        :param first: A vector array with the first element for each dimension.
            The first element is the smallest element if step is positive, and the largest when step is negative.
        :param center: A vector array with the central value for each dimension. The center value is that at the central
            index in the grid, rounded up if center_at_index==True, or the average of the surrounding values when False.
            The center value defaults to 0 if neither first nor last is specified.
        :param last: A vector array with the last element for each dimension. Note that unless include_last is set to
            True for the associated dimension, all but the last element is returned when calling self[axis]!
        :param include_last: A boolean vector array indicating whether the returned vectors, self[axis], should include
            the last element (True) or the penultimate element (default == False). When the step size is not specified,
            it is determined so that the step * (shape - 1) == extent.
        :param ndim: A scalar integer indicating the number of dimensions of the sampling space.
        :param flat: A boolean vector array indicating whether the returned vectors, self[axis], should be
            flattened (True) or returned as an open grid (False)
        :param origin_at_center: A boolean vector array indicating whether the origin should be fft-shifted (True)
            or be ifftshifted to the front (False) of the returned vectors for self[axis].
        :param center_at_index: A boolean vector array indicating whether the center of an even dimension is included in
            the grid. When left as False and the shape has an even number of elements in the corresponding dimension,
            then the next index is used as the center, (self.shape / 2).astype(int). When set to True and the number of
            elements in the corresponding dimension is even, then the center value is not included, only its preceding
            and following elements.
        """
        # Convert all input arguments to vectors of length ndim
        if ndim is None:
            ndim = 1
        shape, step, extent, first, center, last, flat, origin_at_center, include_last, center_at_index, _ = \
            self.__all_to_ndim(shape, step, extent, first, center, last, flat, origin_at_center,
                               include_last, center_at_index, jnp.zeros(ndim))

        def isdef(_):
            return jnp.isnan(_[0])

        if isdef(shape):
            if isdef(extent):
                if isdef(step):
                    step = jnp.ones(shape=shape.shape)
                # step is known
                if isdef(last):
                    if isdef(first):
                        if isdef(1):
                            center = jnp.zeros(shape=shape.shape)
                        # only center and step are known, assume shape == 1
                        first = center
                    # first and step are known
                    if isdef(center):
                        center = (first + 1 / step) % step - step / 2  # Pick the step that is closest to 0
                    # center, first, and step are known
                    shape = 2 * (center - first) / step - (1 - center_at_index)  # Round up to even shape in case of center_at_index
                else:  # last and step are known
                    if isdef(first):
                        if isdef(center):
                            center = (last + 1 / step) % step - step / 2  # Pick the step that is closest to 0
                        # center, last, and step are known
                        shape = 2 * (last - center) / step + include_last - (1 - center_at_index)  # Round up to even shape if center_at_index
                    else:  # first, last, and step are known
                        shape = (last - first) / step + include_last
                # shape is known
            else:  # extent is known
                if isdef(step):
                    step = extent
                # step is known
                extent = jnp.sign(step) * jnp.abs(extent)  # Fix sign of extent if it does not agree with that of step
                shape = extent / step
            # shape is known
        # The shape is known
        shape = jnp.maximum(1, jnp.ceil(shape).astype(int))  # Make sure that the shape is integer and at least 1

        if isdef(step):
            if isdef(extent):
                if isdef(last):
                    if isdef(first):  # Only (potentially) center and shape are known, assume step = 1
                        step = jnp.ones(shape=shape.shape)
                    else:
                        # first and shape are known
                        if isdef(center):  # assume step == 1
                            step = jnp.ones(shape=shape.shape)
                        else:  # center, first, and shape are known
                            step = (center - first) / (shape // 2 * center_at_index + (shape - 1) / 2 * (1 - center_at_index))
                            if (center - first).dtype != step.dtype and jnp.allclose(step, jnp.round(step)):
                                step = step.astype(center.dtype)
                        # step is known
                    # step is known
                else:  # last and shape are known
                    if isdef(first):
                        if isdef(center):  # assume step == 1
                            step = jnp.ones(shape=shape.shape)
                        else:
                            # center is known
                            step = (last - center) / (shape - include_last - shape // 2 * center_at_index - (shape - 1) / 2 * (1 - center_at_index))
                            if (last - center).dtype != step.dtype and jnp.allclose(step, jnp.round(step)):
                                step = step.astype(center.dtype)
                        # step is known
                    else:  # first, last, and shape are known
                        step = (last - first) / (shape - include_last) if jnp.all(shape > include_last) else jnp.ones(1)
                        if (last - first).dtype != step.dtype and jnp.allclose(step, jnp.round(step)):
                            step = step.astype(first.dtype)
                    # step is known
            else:  # extent is known
                step = extent / shape
            if jnp.all(step == step.astype(int)):
                step = step.astype(int)
        # step and shape are known

        if isdef(center):
            if isdef(first):
                if isdef(last):
                    center = jnp.zeros(shape=shape.shape)
                else:  # last is known
                    center = last - step * (shape - include_last - shape // 2 * center_at_index - (shape - 1) / 2 * (1 - center_at_index))
                    if (last - step).dtype != center.dtype and jnp.allclose(center, jnp.round(center)):
                        center = center.astype(step.dtype)
                # center is known
            else:  # first is known
                center = first + (shape // 2 * center_at_index + (shape - 1) / 2 * (1 - center_at_index)) * step
                if (first + step).dtype != center.dtype and jnp.allclose(center, jnp.round(center)):
                    center = center.astype(step.dtype)
        # At this point center, step, and shape are known

        self._shape = shape
        dtype = (step[0] + center[0]).dtype
        self._step = step.astype(dtype)
        self._center = center.astype(dtype)
        self._flat = flat
        self._origin_at_center = origin_at_center
        self.__center_at_index = center_at_index

    @staticmethod
    def from_ranges(*ranges: Union[int, float, complex, Sequence, jax.Array]) -> Grid:
        """
        Converts one or more ranges of numbers to a single Grid object representation.
        The ranges can be specified as separate parameters or as a tuple.

        :param ranges: one or more ranges of uniformly spaced numbers.

        :return: A Grid object that represents the same ranges.
        """
        # Convert slices to range vectors. This won't work with infinite slices
        ranges = [(jnp.arange(rng.start, rng.stop, rng.step) if isinstance(rng, slice) else rng) for rng in ranges]
        ranges = [jnp.array([rng] if jnp.isscalar(rng) else rng) for rng in ranges]  # Treat a scalar as a singleton vector
        if any(_.size < 1 for _ in ranges):
            raise AttributeError('All input ranges should have at least one element.')
        ranges = [(rng.swapaxes(0, axis).reshape(rng.shape[axis], -1)[:, 0] if rng.ndim > 1 else rng)
                  for axis, rng in zip(range(-len(ranges), 0), ranges)]
        # Work out some properties about the shape and the size of each dimension
        shape = jnp.array([rng.size for rng in ranges])
        singleton = shape <= 1
        odd = jnp.mod(shape, 2) == 1
        # Work our what are the first and last elements, which could be at the center
        first = jnp.array([rng[0] for rng in ranges])  # first when fftshifted, center+ otherwise
        before_center = jnp.array([rng[int((rng.size - 1) / 2)] for rng in ranges])  # last when ifftshifted, center+ otherwise
        after_center = jnp.array([rng[-int(rng.size / 2)] for rng in ranges])  # first when ifftshifted, center- otherwise
        last = jnp.array([rng[-1] for rng in ranges])  # last when fftshifted, center- otherwise
        # The last value is included!

        # If it is not monotonous, it is ifftshifted
        origin_at_center = jnp.abs(last - first) >= jnp.abs(before_center - after_center)
        # Figure out what is the step size and the center element
        extent_m1 = origin_at_center * (last - first) + (1 - origin_at_center) * (before_center - after_center)
        step = extent_m1 / (shape - 1 + singleton)  # Note that the step can be a complex number
        center = origin_at_center * (odd * before_center + (1 - odd) * after_center) + (1 - origin_at_center) * first

        return Grid(shape=shape, step=step, center=center, flat=False, origin_at_center=origin_at_center)

    #
    # Grid and array properties
    #
    @property
    def ndim(self) -> int:
        """The number of dimensions of the space this grid spans."""
        return self.shape.size

    @property
    def shape(self) -> jax.Array:
        """The number of sample points along each axis of the grid."""
        return self._shape

    @property
    def step(self) -> jax.Array:
        """The sample spacing along each axis of the grid."""
        return self._step

    @property
    def center(self) -> jax.Array:
        """The central coordinate of the grid."""
        return self._center

    @property
    def center_at_index(self) -> jax.Array:
        """
        Boolean vector indicating whether the central coordinate is aligned with a grid point when the number
        of points is even along the associated axis. This has no effect when the the number of sample points is odd.
        """
        return self.__center_at_index

    @property
    def flat(self) -> jax.Array:
        """
        Boolean vector indicating whether self[axis] returns flattened (raveled) vectors (True) or not (False).
        """
        return self._flat

    @property
    def origin_at_center(self) -> jax.Array:
        """
        Boolean vector indicating whether self[axis] returns ranges that are monotonous (True) or
        ifftshifted so that the central index is the first element of the sequence (False).
        """
        return self._origin_at_center

    #
    # Conversion methods
    #

    @property
    def as_flat(self) -> Grid:
        """
        :return: A new Grid object where all the ranges are 1d-vectors (flattened or raveled)
        """
        shape, step, center, center_at_index, origin_at_center = \
            self.shape, self.step, self.center, self.center_at_index, self.origin_at_center
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=True, origin_at_center=origin_at_center)

    @property
    def as_non_flat(self) -> Grid:
        """
        :return: A new Grid object where all the ranges are 1d-vectors (flattened or raveled)
        """
        shape, step, center, center_at_index, origin_at_center = \
            self.shape, self.step, self.center, self.center_at_index, self.origin_at_center
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=False, origin_at_center=origin_at_center)

    @property
    def as_origin_at_0(self) -> Grid:
        """
        :return: A new Grid object where all the ranges are ifftshifted so that the origin as at index 0.
        """
        shape, step, center, center_at_index, flat = self.shape, self.step, self.center, self.center_at_index, self.flat
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=flat, origin_at_center=False)

    @property
    def as_origin_at_center(self) -> Grid:
        """
        :return: A new Grid object where all the ranges have the origin at the center index, even when the number of elements is odd.
        """
        shape, step, center, center_at_index, flat = self.shape, self.step, self.center, self.center_at_index, self.flat
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=flat, origin_at_center=True)

    def swapaxes(self, axes: Union[slice, Sequence, jax.Array]) -> Grid:
        """Reverses the order of the specified axes."""
        axes = jnp.array(axes).flatten()
        all_axes = jnp.arange(self.ndim)
        all_axes[axes] = axes[::-1]
        return self.transpose(all_axes)

    def transpose(self, axes: Union[None, slice, Sequence, jax.Array]=None) -> Grid:
        """Reverses the order of all axes."""
        if axes is None:
            axes = jnp.arange(self.ndim-1, -1, -1)
        return self.project(axes)

    def project(self, axes_to_keep: Union[int, slice, Sequence, jax.Array, None] = None,
                axes_to_remove: Union[int, slice, Sequence, jax.Array, None] = None) -> Grid:
        """
        Removes all but the specified axes and reduces the dimensions to the number of specified axes.

        :param axes_to_keep: The indices of the axes to keep.
        :param axes_to_remove: The indices of the axes to remove. Default: None

        :return: A Grid object with ndim == len(axes) and shape == shape[axes].
        """
        if axes_to_keep is None:
            axes_to_keep = jnp.arange(self.ndim)
        elif isinstance(axes_to_keep, slice):
            axes_to_keep = jnp.arange(self.ndim)[axes_to_keep]
        if jnp.isscalar(axes_to_keep):
            axes_to_keep = [axes_to_keep]
        axes_to_keep = jnp.array(axes_to_keep)
        if axes_to_remove is None:
            axes_to_remove = []
        elif isinstance(axes_to_remove, slice):
            axes_to_remove = jnp.arange(self.ndim)[axes_to_remove]
        if jnp.isscalar(axes_to_remove):
            axes_to_remove = [axes_to_remove]
        # Do some checks
        if jnp.any(axes_to_keep >= self.ndim) or jnp.any(axes_to_keep < -self.ndim):
            raise IndexError(f"Axis range {axes_to_keep} requested from a Grid of dimension {self.ndim}.")
        # Make sure that the axes are non-negative
        axes_to_keep = [_ % self.ndim for _ in axes_to_keep]
        axes_to_remove = [_ % self.ndim for _ in axes_to_remove]
        axes_to_keep = jnp.array([_ for _ in axes_to_keep if _ not in axes_to_remove])

        if len(axes_to_keep) > 0:
            return Grid(shape=self.shape[axes_to_keep], step=self.step[axes_to_keep], center=self.center[axes_to_keep],
                        flat=self.flat[axes_to_keep],
                        origin_at_center=self.origin_at_center[axes_to_keep],
                        center_at_index=self.center_at_index[axes_to_keep]
                        )
        else:
            return Grid([])

    #
    # Derived properties
    #

    @property
    def first(self) -> jnp.ndarray:
        """A vector with the first element of each range."""
        center_is_not_at_index = ~self.center_at_index & (self.shape % 2 == 0)
        result = self._center - self.step * (self.shape // 2)
        if jnp.any(center_is_not_at_index):
            half_step = self.step // 2 if jnp.all(self.step % 2 == 0) else self.step / 2
            result = result + center_is_not_at_index * half_step
        return result

    @property
    def extent(self) -> jnp.ndarray:
        """ The spatial extent of the sampling grid."""
        return self.shape * self.step

    #
    # Sequence methods
    #

    @property
    def size(self) -> jax.Array:
        """ The total number of sampling points as an integer scalar. """
        return jnp.prod(self.shape)

    @property
    def dtype(self):
        """ The numeric data type for the coordinates. """
        return (self.step[0] + self.center[0]).dtype

    #
    # Frequency grids
    #

    @property
    def f(self) -> Grid:
        """ The equivalent frequency Grid. """
        shape, step, flat = self.shape, 1 / self.extent, self.flat

        return Grid(shape=shape, step=step, flat=flat, origin_at_center=False, center_at_index=True)

    @property
    def k(self) -> Grid:
        """ The equivalent k-space Grid. """
        return self.f * (2 * jnp.pi)

    #
    # Arithmetic methods
    #
    def __add__(self, term) -> Grid:
        """ Add a scalar or vector offset to the Grid coordinates. """
        d = self.__dict__
        new_center = self.center + jnp.asarray(term)
        d['center'] = new_center
        return Grid(**d)

    def __mul__(self, factor: Union[int, float, complex, Sequence, jax.Array]) -> Grid:
        """
        Scales all ranges with a factor.

        :param factor: A scalar factor for all dimensions, or a vector of factors, one for each dimension.

        :return: A new scaled Grid object.
        """
        if isinstance(factor, Grid):
            raise TypeError("A Grid object can't be multiplied with a Grid object."
                            + "Use matmul @ to determine the tensor space.")
        d = self.__dict__
        factor = jnp.asarray(factor)
        new_step = self.step * factor
        new_center = self.center * factor
        d['step'] = new_step
        d['center'] = new_center
        return Grid(**d)

    def __rmul__(self, factor: Union[int, float, complex, Sequence, jax.Array]) -> Grid:
        """
        Scales all ranges with a factor.

        :param factor: A scalar factor for all dimensions, or a vector of factors, one for each dimension.

        :return: A new scaled Grid object.
        """
        return self * factor  # Scalars commute.

    def __matmul__(self, other: Grid) -> Grid:
        """
        Determines the Grid spanning the tensor space, with ndim equal to the sum of both ndims.

        :param other: The Grid with the right-hand dimensions.

        :return: A new Grid with ndim == self.ndim + other.ndim.
        """
        return Grid(shape=(*self.shape, *other.shape), step=(*self.step, *other.step),
                    center=(*self.center, *other.center),
                    flat=(*self.flat, *other.flat),
                    origin_at_center=(*self.origin_at_center, *other.origin_at_center),
                    center_at_index=(*self.center_at_index, *other.center_at_index)
                    )

    def __sub__(self, term: Union[int, float, complex, Sequence, jnp.ndarray]) -> Grid:
        """ Subtract a (scalar) value from all Grid coordinates. """
        return self + (- term)

    def __truediv__(self, denominator: Union[int, float, complex, Sequence, jnp.ndarray]) -> Grid:
        """
        Divide the grid coordinates by a value.

        :param denominator: The denominator to divide by.

        :return: A new Grid with the divided coordinates.
        """
        return self * (1 / denominator)

    def __neg__(self):
        """ Invert the coordinate values and the direction of the axes. """
        return self.__mul__(-1)

    #
    # iterator methods
    #

    def __len__(self) -> int:
        """
        The number of axes in this sampling grid.
        """
        return self.ndim

    def __getitem__(self, key: Union[int, slice, Sequence]):
        """
        Select one or more axes from a multi-dimensional grid.
        """
        scalar_key = jnp.isscalar(key)
        indices = jnp.atleast_1d(jnp.arange(self.ndim)[key])
        result = []
        for axis in indices.ravel():
            try:
                c, st, sh = self.center[axis], self.step[axis], self.shape[axis]
            except IndexError:
                raise IndexError(f"Axis range {axis} requested from a Grid of dimension {self.ndim}.")
            rng = jnp.arange(sh) - (sh // 2)
            if sh > 1:  # Define the center as 0 to avoid trouble with * jnp.inf when this is a singleton dimension.
                rng = rng * st
            rng = rng + c
            if not self.__center_at_index[axis] and (sh % 2 == 0):
                rng = rng + st / 2
            if not self._origin_at_center[axis]:
                rng = jnp.fft.ifftshift(rng)
            if not self.flat[axis]:
                rng = dim.to_axis(rng, axis=axis, ndim=self.ndim)

            result.append(rng)

        if scalar_key:
            return result[0]  # Unpack again
        return tuple(result)

    def __iter__(self):
        for idx in range(self.ndim):
            yield self[idx]

    #
    # General object properties
    #

    @property
    def __dict__(self):
        shape, step, center, flat, center_at_index, origin_at_center = \
            self.shape, self.step, self.center, self.flat, self.center_at_index, self.origin_at_center
        return dict(shape=shape, step=step, center=center, flat=flat,
                    center_at_index=center_at_index, origin_at_center=origin_at_center)

    @property
    def immutable(self) -> Grid:
        """Return a new immutable Grid object. """
        return Grid(**self.__dict__)

    def __str__(self) -> str:
        core_props = self.__dict__.copy()
        arg_desc = ','.join([f'{k}={str(v)}' for k, v in core_props.items()])
        return f"{type(self).__name__}({arg_desc:s})"

    def __repr__(self) -> str:
        core_props = self.__dict__.copy()
        core_props['dtype'] = self.dtype
        arg_desc = ','.join([f'{k}={repr(v)}' for k, v in core_props.items()])
        return f"{type(self).__name__}({arg_desc:s})"

    def __eq__(self, other: Grid) -> bool:
        """ Compares two Grid objects. """
        result = (
                type(self) == type(other) and
                (self.shape == other.shape).all() and
                (self.step == other.step).all() and
                (self.center == other.center).all() and
                (self.flat == other.flat).all() and
                (self.center_at_index == other.center_at_index).all() and
                (self.origin_at_center == other.origin_at_center).all() and
                self.dtype == other.dtype
        )
        return result

    def __hash__(self) -> int:
        return hash(repr(self))

    #
    # Protected and private methods
    #

    @staticmethod
    def __all_to_ndim(*args):
        """
        Helper method to ensures that all arguments are all numpy vectors of the same length, self.ndim.
        """
        return jnp.broadcast_arrays(*(jnp.nan if _ is None else jnp.array(_).ravel() for _ in args))

    def _to_ndim(self, arg) -> jax.Array:
        """Helper method to ensure that all arguments are all jax.numpy vectors of the same length, self.ndim."""
        return self.__all_to_ndim(self.shape, arg)[-1]

    def tree_flatten(self):
        """Method required for JAX serialization."""
        return self.__dict__.items(), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Method required for JAX deserialization."""
        return cls(**dict(children))
