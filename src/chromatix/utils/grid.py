from __future__ import annotations

from typing import Union, Sequence
import numpy as np
import warnings

from macromax.utils import dim


class Grid(Sequence):
    """
    A class representing an immutable uniformly-spaced plaid Cartesian grid and its Fourier Transform.
    Unlike the MutableGrid, objects of this class cannot be changed after creation.

    See also :class:`MutableGrid`
    """
    def __init__(self, shape=None, step=None, *, extent=None, first=None, center=None, last=None, include_last=False,
                 ndim: int = None,
                 flat: Union[bool, Sequence, np.ndarray] = False,
                 origin_at_center: Union[bool, Sequence, np.ndarray] = True,
                 center_at_index: Union[bool, Sequence, np.ndarray] = True):
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
        # Figure out what dimension is required
        if ndim is None:
            ndim = 0
            if shape is not None:
                ndim = np.maximum(ndim, np.array(shape).size)
            if step is not None:
                ndim = np.maximum(ndim, np.array(step).size)
            if extent is not None:
                ndim = np.maximum(ndim, np.array(extent).size)
            if first is not None:
                ndim = np.maximum(ndim, np.array(first).size)
            if center is not None:
                ndim = np.maximum(ndim, np.array(center).size)
            if last is not None:
                ndim = np.maximum(ndim, np.array(last).size)
        self.__ndim = ndim

        def is_vector(value):
            return value is not None and not np.isscalar(value)
        self.__multidimensional = is_vector(shape) or is_vector(step) or is_vector(extent) or \
                                  is_vector(first) or is_vector(center) or is_vector(last)

        # Convert all input arguments to vectors of length ndim
        shape, step, extent, first, center, last, flat, origin_at_center, include_last, center_at_index = \
            self.__all_to_ndim(shape, step, extent, first, center, last, flat, origin_at_center,
                               include_last, center_at_index)

        if shape is None:
            if extent is None:
                if step is None:
                    step = 1
                # step is known
                if last is None:
                    if first is None:
                        if center is None:
                            center = self._to_ndim(0)
                        # only center and step are known, assume shape == 1
                        first = center
                    # first and step are known
                    if center is None:
                        center = (first + 1 / step) % step - step / 2  # Pick the step that is closest to 0
                    # center, first, and step are known
                    shape = 2 * (center - first) / step - (1 - center_at_index)  # Round up to even shape in case of center_at_index
                else:  # last and step are known
                    if first is None:
                        if center is None:
                            center = (last + 1 / step) % step - step / 2  # Pick the step that is closest to 0
                        # center, last, and step are known
                        shape = 2 * (last - center) / step + include_last - (1 - center_at_index)  # Round up to even shape if center_at_index
                    else:  # first, last, and step are known
                        shape = (last - first) / step + include_last
                # shape is known
            else:  # extent is known
                if step is None:
                    step = extent
                # step is known
                extent = np.sign(step) * np.abs(extent)  # Fix sign of extent if it does not agree with that of step
                shape = extent / step
            # shape is known
        # The shape is known
        shape = np.maximum(1, np.ceil(shape).astype(int))  # Make sure that the shape is integer and at least 1

        if step is None:
            if extent is None:
                if last is None:
                    if first is None:  # Only (potentially) center and shape are known, assume step = 1
                        step = self._to_ndim(1)
                    else:
                        # first and shape are known
                        if center is None:  # assume step == 1
                            step = self._to_ndim(1)
                        else:  # center, first, and shape are known
                            step = (center - first) / (shape // 2 * center_at_index + (shape - 1) / 2 * (1 - center_at_index))
                            if (center - first).dtype != step.dtype and np.allclose(step, np.round(step)):
                                step = step.astype(center.dtype)
                        # step is known
                    # step is known
                else:  # last and shape are known
                    if first is None:
                        if center is None:  # assume step == 1
                            step = self._to_ndim(1)
                        else:
                            # center is known
                            step = (last - center) / (shape - include_last - shape // 2 * center_at_index - (shape - 1) / 2 * (1 - center_at_index))
                            if (last - center).dtype != step.dtype and np.allclose(step, np.round(step)):
                                step = step.astype(center.dtype)
                        # step is known
                    else:  # first, last, and shape are known
                        step = (last - first) / (shape - include_last) if np.all(shape > include_last) else np.ones(1)
                        if (last - first).dtype != step.dtype and np.allclose(step, np.round(step)):
                            step = step.astype(first.dtype)
                    # step is known
            else:  # extent is known
                step = extent / shape
            if np.all(step == step.astype(int)):
                step = step.astype(int)
        # step and shape are known

        if center is None:
            if first is None:
                if last is None:
                    center = self._to_ndim(0)
                else:  # last is known
                    center = last - step * (shape - include_last - shape // 2 * center_at_index - (shape - 1) / 2 * (1 - center_at_index))
                    if (last - step).dtype != center.dtype and np.allclose(center, np.round(center)):
                        center = center.astype(step.dtype)
                # center is known
            else:  # first is known
                center = first + (shape // 2 * center_at_index + (shape - 1) / 2 * (1 - center_at_index)) * step
                if (first + step).dtype != center.dtype and np.allclose(center, np.round(center)):
                    center = center.astype(step.dtype)
        # center, step, and shape are known now

        # Some sanity checks
        if extent is not None and not np.allclose(extent / step, shape) and np.any(np.maximum(1, np.ceil(extent / step)) != shape):
            raise ValueError(f"Extent {extent} and step {step} are not compatible with shape {shape} because extent / step = {extent / step} != {shape} = shape.")
        if last is not None and first is not None and np.any(shape * step != last + step * include_last - first):
            raise ValueError(f"First={first} and last={last} (include_last={include_last}) do not correspond to a step {step} and shape {shape}.")
        if center is not None and first is not None and np.any(center != first + (shape // 2 * center_at_index + (shape - 1) / 2 * (1 - center_at_index)) * step):
            raise ValueError(f"First={first} and center={center} do not correspond to a step {step} and shape {shape} (center_at_index={center_at_index}).")

        if np.any(shape < 1):
            warnings.warn(f'shape = {shape}. All input ranges should have at least one element.')
            shape = np.maximum(1, shape)

        self._shape = shape
        dtype = (step[0] + center[0]).dtype if self.ndim > 0 else float
        self._step = step.astype(dtype)
        self._center = center.astype(dtype)
        self._flat = flat
        self._origin_at_center = origin_at_center
        self.__center_at_index = center_at_index

    @staticmethod
    def from_ranges(*ranges: Union[int, float, complex, Sequence, np.ndarray]) -> Grid:
        """
        Converts one or more ranges of numbers to a single Grid object representation.
        The ranges can be specified as separate parameters or as a tuple.

        :param ranges: one or more ranges of uniformly spaced numbers.

        :return: A Grid object that represents the same ranges.
        """
        # Convert slices to range vectors. This won't work with infinite slices
        ranges = [(np.arange(rng.start, rng.stop, rng.step) if isinstance(rng, slice) else rng) for rng in ranges]
        ranges = [np.array([rng] if np.isscalar(rng) else rng) for rng in ranges]  # Treat a scalar as a singleton vector
        if any(_.size < 1 for _ in ranges):
            raise AttributeError('All input ranges should have at least one element.')
        ranges = [(rng.swapaxes(0, axis).reshape(rng.shape[axis], -1)[:, 0] if rng.ndim > 1 else rng)
                  for axis, rng in zip(range(-len(ranges), 0), ranges)]
        # Work out some properties about the shape and the size of each dimension
        shape = np.array([rng.size for rng in ranges])
        singleton = shape <= 1
        odd = np.mod(shape, 2) == 1
        # Work our what are the first and last elements, which could be at the center
        first = np.array([rng[0] for rng in ranges])  # first when fftshifted, center+ otherwise
        before_center = np.array([rng[int((rng.size - 1) / 2)] for rng in ranges])  # last when ifftshifted, center+ otherwise
        after_center = np.array([rng[-int(rng.size / 2)] for rng in ranges])  # first when ifftshifted, center- otherwise
        last = np.array([rng[-1] for rng in ranges])  # last when fftshifted, center- otherwise
        # The last value is included!

        # If it is not monotonous, it is ifftshifted
        origin_at_center = np.abs(last - first) >= np.abs(before_center - after_center)
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
        return self.__ndim

    @property
    def shape(self) -> np.array:
        """The number of sample points along each axis of the grid."""
        return self._shape.copy()

    @property
    def step(self) -> np.ndarray:
        """The sample spacing along each axis of the grid."""
        return self._step.copy()

    @property
    def center(self) -> np.ndarray:
        """The central coordinate of the grid."""
        return self._center.copy()

    @property
    def center_at_index(self) -> np.array:
        """
        Boolean vector indicating whether the central coordinate is aligned with a grid point when the number
        of points is even along the associated axis. This has no effect when the the number of sample points is odd.
        """
        return self.__center_at_index.copy()

    @property
    def flat(self) -> np.array:
        """
        Boolean vector indicating whether self[axis] returns flattened (raveled) vectors (True) or not (False).
        """
        return self._flat.copy()

    @property
    def origin_at_center(self) -> np.array:
        """
        Boolean vector indicating whether self[axis] returns ranges that are monotonous (True) or
        ifftshifted so that the central index is the first element of the sequence (False).
        """
        return self._origin_at_center.copy()

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
        if not self.multidimensional:
            shape, step, center, center_at_index, origin_at_center = \
                shape[0], step[0], center[0], center_at_index[0], origin_at_center[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=True, origin_at_center=origin_at_center)

    @property
    def as_non_flat(self) -> Grid:
        """
        :return: A new Grid object where all the ranges are 1d-vectors (flattened or raveled)
        """
        shape, step, center, center_at_index, origin_at_center = \
            self.shape, self.step, self.center, self.center_at_index, self.origin_at_center
        if not self.multidimensional:
            shape, step, center, center_at_index, origin_at_center = \
                shape[0], step[0], center[0], center_at_index[0], origin_at_center[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=False, origin_at_center=origin_at_center)

    @property
    def as_origin_at_0(self) -> Grid:
        """
        :return: A new Grid object where all the ranges are ifftshifted so that the origin as at index 0.
        """
        shape, step, center, center_at_index, flat = self.shape, self.step, self.center, self.center_at_index, self.flat
        if not self.multidimensional:
            shape, step, center, center_at_index, flat = shape[0], step[0], center[0], center_at_index[0], flat[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=flat, origin_at_center=False)

    @property
    def as_origin_at_center(self) -> Grid:
        """
        :return: A new Grid object where all the ranges have the origin at the center index, even when the number of elements is odd.
        """
        shape, step, center, center_at_index, flat = self.shape, self.step, self.center, self.center_at_index, self.flat
        if not self.multidimensional:
            shape, step, center, center_at_index, flat = shape[0], step[0], center[0], center_at_index[0], flat[0]
        return Grid(shape=shape, step=step, center=center, center_at_index=center_at_index,
                    flat=flat, origin_at_center=True)

    def swapaxes(self, axes: Union[slice, Sequence, np.array]) -> Grid:
        """Reverses the order of the specified axes."""
        axes = np.array(axes).flatten()
        all_axes = np.arange(self.ndim)
        all_axes[axes] = axes[::-1]
        return self.transpose(all_axes)

    def transpose(self, axes: Union[None, slice, Sequence, np.array]=None) -> Grid:
        """Reverses the order of all axes."""
        if axes is None:
            axes = np.arange(self.ndim-1, -1, -1)
        return self.project(axes)

    def project(self, axes_to_keep: Union[int, slice, Sequence, np.array, None] = None,
                axes_to_remove: Union[int, slice, Sequence, np.array, None] = None) -> Grid:
        """
        Removes all but the specified axes and reduces the dimensions to the number of specified axes.

        :param axes_to_keep: The indices of the axes to keep.
        :param axes_to_remove: The indices of the axes to remove. Default: None

        :return: A Grid object with ndim == len(axes) and shape == shape[axes].
        """
        if axes_to_keep is None:
            axes_to_keep = np.arange(self.ndim)
        elif isinstance(axes_to_keep, slice):
            axes_to_keep = np.arange(self.ndim)[axes_to_keep]
        if np.isscalar(axes_to_keep):
            axes_to_keep = [axes_to_keep]
        axes_to_keep = np.array(axes_to_keep)
        if axes_to_remove is None:
            axes_to_remove = []
        elif isinstance(axes_to_remove, slice):
            axes_to_remove = np.arange(self.ndim)[axes_to_remove]
        if np.isscalar(axes_to_remove):
            axes_to_remove = [axes_to_remove]
        # Do some checks
        if np.any(axes_to_keep >= self.ndim) or np.any(axes_to_keep < -self.ndim):
            raise IndexError(f"Axis range {axes_to_keep} requested from a Grid of dimension {self.ndim}.")
        # Make sure that the axes are non-negative
        axes_to_keep = [_ % self.ndim for _ in axes_to_keep]
        axes_to_remove = [_ % self.ndim for _ in axes_to_remove]
        axes_to_keep = np.array([_ for _ in axes_to_keep if _ not in axes_to_remove])

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
    def first(self) -> np.ndarray:
        """A vector with the first element of each range."""
        center_is_not_at_index = ~self.center_at_index & (self.shape % 2 == 0)
        result = self._center - self.step * (self.shape // 2)
        if np.any(center_is_not_at_index):
            half_step = self.step // 2 if np.all(self.step % 2 == 0) else self.step / 2
            result = result + center_is_not_at_index * half_step
        return result

    @property
    def extent(self) -> np.ndarray:
        """ The spatial extent of the sampling grid."""
        return self.shape * self.step

    #
    # Sequence methods
    #

    @property
    def size(self) -> int:
        """ The total number of sampling points as an integer scalar. """
        return int(np.prod(self.shape))

    @property
    def dtype(self):
        """ The numeric data type for the coordinates. """
        return (self.step[0] + self.center[0]).dtype if self.ndim > 0 else float

    #
    # Frequency grids
    #

    @property
    def f(self) -> Grid:
        """ The equivalent frequency Grid. """
        with np.errstate(divide='ignore'):
            shape, step, flat = self.shape, 1 / self.extent, self.flat
        if not self.multidimensional:
            shape, step, flat = shape[0], step[0], flat[0]

        return Grid(shape=shape, step=step, flat=flat, origin_at_center=False, center_at_index=True)

    @property
    def k(self) -> Grid:
        """ The equivalent k-space Grid. """
        return self.f * (2 * np.pi)

    #
    # Arithmetic methods
    #
    def __add__(self, term) -> Grid:
        """ Add a scalar or vector offset to the Grid coordinates. """
        d = self.__dict__
        new_center = self.center + np.asarray(term)
        if not self.multidimensional:
            new_center = new_center[0]
        d['center'] = new_center
        return Grid(**d)

    def __mul__(self, factor: Union[int, float, complex, Sequence, np.array]) -> Grid:
        """
        Scales all ranges with a factor.

        :param factor: A scalar factor for all dimensions, or a vector of factors, one for each dimension.

        :return: A new scaled Grid object.
        """
        if isinstance(factor, Grid):
            raise TypeError("A Grid object can't be multiplied with a Grid object."
                            + "Use matmul @ to determine the tensor space.")
        d = self.__dict__
        factor = np.asarray(factor)
        new_step = self.step * factor
        new_center = self.center * factor
        if not self.multidimensional:
            new_step = new_step[0]
            new_center = new_center[0]
        d['step'] = new_step
        d['center'] = new_center
        return Grid(**d)

    def __rmul__(self, factor: Union[int, float, complex, Sequence, np.array]) -> Grid:
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

    def __sub__(self, term: Union[int, float, complex, Sequence, np.ndarray]) -> Grid:
        """ Subtract a (scalar) value from all Grid coordinates. """
        return self + (- term)

    def __truediv__(self, denominator: Union[int, float, complex, Sequence, np.ndarray]) -> Grid:
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
        Or, the number of elements when this object is not multi-dimensional.
        """
        if self.multidimensional:
            return self.ndim
        else:
            return self.shape[0]  # Behave as a single Sequence

    def __getitem__(self, key: Union[int, slice, Sequence]):
        """
        Select one or more axes from a multi-dimensional grid,
        or select elements from a single-dimensional object.
        """
        scalar_key = np.isscalar(key)
        indices = np.atleast_1d(np.arange(self.ndim if self.multidimensional else self.shape[0])[key])
        result = []
        for idx in indices.ravel():
            axis = idx if self.multidimensional else 0  # Behave as a single Sequence

            try:
                c, st, sh = self.center[axis], self.step[axis], self.shape[axis]
            except IndexError as err:
                raise IndexError(f"Axis range {axis} requested from a Grid of dimension {self.ndim}.")
            rng = np.arange(sh) - (sh // 2)
            if sh > 1:  # Define the center as 0 to avoid trouble with * np.inf when this is a singleton dimension.
                rng = rng * st
            rng = rng + c
            if not self.__center_at_index[axis] and (sh % 2 == 0):
                rng = rng + st / 2
            if not self._origin_at_center[axis]:
                rng = np.fft.ifftshift(rng)  # Not loading the whole fft library just for this!
            if not self.flat[axis]:
                rng = dim.to_axis(rng, axis=axis, ndim=self.ndim)

            result.append(rng if self.multidimensional else rng[idx])

        if scalar_key:
            result = result[0]  # Unpack again

        return result

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    #
    # General object properties
    #

    @property
    def __dict__(self):
        shape, step, center, flat, center_at_index, origin_at_center = \
            self.shape, self.step, self.center, self.flat, self.center_at_index, self.origin_at_center
        if not self.multidimensional:
            shape, step, center, flat, center_at_index, origin_at_center = \
                shape[0], step[0], center[0], flat[0], center_at_index[0], origin_at_center[0]
        return dict(shape=shape, step=step, center=center, flat=flat,
                    center_at_index=center_at_index, origin_at_center=origin_at_center)

    @property
    def immutable(self) -> Grid:
        """Return a new immutable Grid object. """
        return Grid(**self.__dict__)

    @property
    def mutable(self) -> MutableGrid:
        """Return a new MutableGrid object. """
        return MutableGrid(**self.__dict__)

    def __str__(self) -> str:
        core_props = self.__dict__.copy()
        arg_desc = ','.join([f'{k}={str(v)}' for k, v in core_props.items()])
        return f"{type(self).__name__}({arg_desc:s})"

    def __repr__(self) -> str:
        core_props = self.__dict__.copy()
        core_props['dtype'] = self.dtype
        # core_props['multidimensional'] = self.multidimensional
        arg_desc = ','.join([f'{k}={repr(v)}' for k, v in core_props.items()])
        return f"{type(self).__name__}({arg_desc:s})"

    # def __format__(self, format_spec: str = "") -> str:
    #     return f"{type(self).__name__}({tuple(self.shape)}, ({', '.join(format(_, format_spec) for _ in self.step)}), ({', '.join(format(_, format_spec) for _ in self.center)}))"
    #
    def __eq__(self, other: Grid) -> bool:
        """ Compares two Grid objects. """
        return type(self) == type(other) and np.all(self.shape == other.shape) and np.all(self.step == other.step) \
            and np.all(self.center == other.center) and np.all(self.flat == other.flat) \
            and np.all(self.center_at_index == other.center_at_index) \
            and np.all(self.origin_at_center == other.origin_at_center) and self.dtype == other.dtype

    def __hash__(self) -> int:
        return hash(repr(self))

    #
    # Assorted property
    #
    @property
    def multidimensional(self) -> bool:
        """Single-dimensional grids behave as Sequences, multi-dimensional behave as a Sequence of vectors.
        TODO: Remove this feature? It tends to be a source of bugs.
        """
        return self.__multidimensional

    #
    # Protected and private methods
    #

    def _to_ndim(self, arg) -> np.array:
        """
        Helper method to ensure that all arguments are all numpy vectors of the same length, self.ndim.
        """
        if arg is not None:
            arg = np.array(arg).flatten()
            if np.isscalar(arg) or arg.size == 1:
                arg = np.repeat(arg, repeats=self.ndim)
            elif arg.size != self.ndim:
                raise ValueError(
                    f"All input arguments should be scalar or of length {self.ndim}, not {arg.size} as {arg}.")
        return arg

    def __all_to_ndim(self, *args):
        """
        Helper method to ensures that all arguments are all numpy vectors of the same length, self.ndim.
        """
        return tuple([self._to_ndim(arg) for arg in args])


class MutableGrid(Grid):
    """
    A class representing a mutable uniformly-spaced plaid Cartesian grid and its Fourier Transform.

    See also :class:`Grid`
    """
    def __init__(self, shape=None, step=None, extent=None, first=None, center=None, last=None, include_last=False,
                 ndim: int = None,
                 flat: Union[bool, Sequence, np.ndarray] = False,
                 origin_at_center: Union[bool, Sequence, np.ndarray] = True,
                 center_at_index: Union[bool, Sequence, np.ndarray] = True):
        """
        Construct a mutable Grid object.

        :param shape: An integer vector array with the shape of the sampling grid.
        :param step: A vector array with the spacing of the sampling grid.
        :param extent: The extent of the sampling grid as shape * step
        :param first: A vector array with the first element for each dimension.
            The first element is the smallest element if step is positive, and the largest when step is negative.
        :param center: A vector array with the center element for each dimension. The center position in the grid is
            rounded to the next integer index unless center_at_index is set to False for that partical axis.
        :param last: A vector array with the last element for each dimension. Unless include_last is set to True for
            the associated dimension, all but the last element is returned when calling self[axis].
        :param include_last: A boolean vector array indicating whether the returned vectors, self[axis], should include
            the last element (True) or all-but-the-last (False)
        :param ndim: A scalar integer indicating the number of dimensions of the sampling space.
        :param flat: A boolean vector array indicating whether the returned vectors, self[axis], should be
            flattened (True) or returned as an open grid (False)
        :param origin_at_center: A boolean vector array indicating whether the origin should be fft-shifted (True)
            or be ifftshifted to the front (False) of the returned vectors for self[axis].
        :param center_at_index: A boolean vector array indicating whether the center of the grid should be rounded to an
            integer index for each dimension. If False and the shape has an even number of elements, the next index is
            used as the center, (self.shape / 2).astype(int).
        """
        super().__init__(shape=shape, step=step, extent=extent, first=first, center=center, last=last,
                         include_last=include_last, ndim=ndim, flat=flat, origin_at_center=origin_at_center,
                         center_at_index=center_at_index)

    @property
    def shape(self) -> np.array:
        return super().shape

    @shape.setter
    def shape(self, new_shape: Union[int, Sequence, np.array]):
        if new_shape is not None:
            self._shape = self._to_ndim(new_shape)

    @property
    def step(self) -> np.ndarray:
        return super().step

    @step.setter
    def step(self, new_step: Union[int, float, Sequence, np.array]):
        self._step = self._to_ndim(new_step)
        self._center = self._center.astype(self.dtype)

    @property
    def center(self) -> np.ndarray:
        return super().center

    @center.setter
    def center(self, new_center: Union[int, float, Sequence, np.array]):
        self._center = self._to_ndim(new_center).astype(self.dtype)

    @property
    def flat(self) -> np.array:
        return super().flat

    @flat.setter
    def flat(self, value: Union[bool, Sequence, np.array]):
        self._flat = self._to_ndim(value)

    @property
    def origin_at_center(self) -> np.array:
        return super().origin_at_center

    @origin_at_center.setter
    def origin_at_center(self, value: Union[bool, Sequence, np.array]):
        self._origin_at_center = self._to_ndim(value)

    @property
    def first(self) -> np.ndarray:
        """
        :return: A vector with the first element of each range
        """
        return super().first

    @first.setter
    def first(self, new_first: Union[int, float, Sequence, np.ndarray]):
        self._center = super().center + self._to_ndim(new_first) - self.first

    @property
    def dtype(self):
        """ The numeric data type for the coordinates. """
        return (self.step[0] + self.center[0]).dtype

    @dtype.setter
    def dtype(self, new_type: dtype):
        """ Sets the dtype of the range, updating the step and center coordinate."""
        self._step = self._step.astype(new_type)
        self._center = self._center.astype(new_type)

    def __iadd__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.center += np.asarray(number)

    def __imul__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.step *= np.asarray(number)
        self.center *= np.asarray(number)

    def __isub__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.center -= np.asarray(number)

    def __idiv__(self, number: Union[int, float, complex, Sequence, np.ndarray]):
        self.step /= np.asarray(number)
        self.center /= np.asarray(number)
