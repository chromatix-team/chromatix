"""
The module provides the abstract :class:`Bound` to represent the boundary of the simulation, e.g. periodic, or
gradually more absorbing. Specific boundaries are implemented as subclasses and can be used directly as the `bound`
argument to :func:`macromax.solve` or :class:`macromax.Solution`. The precludes the inclusion of boundaries in the
material description.

Code ported from [this file in MacroMax](https://github.com/tttom/MacroMax/blob/master/python/macromax/bound.py) to work
with JAX.
"""
from __future__ import annotations

from typing import Union, Sequence, Callable

import jax
import jax.numpy as jnp

from chromatix.utils import Grid


class Electric:
    """ Mixin for Bound to indicate that the electric susceptibility is non-zero."""
    @property
    def background_permittivity(self) -> Complex:
        """A complex scalar indicating the permittivity of the background."""
        return 0.0

    @property
    def electric_susceptibility(self) -> jax.Array:
        return NotImplemented

    @property
    def permittivity(self) -> jax.Array:
        """
        The electric permittivity, epsilon, at every sample point.
        Note that the returned array may have singleton dimensions that must be broadcast!
        """
        return self.background_permittivity + self.electric_susceptibility


class Bound:
    """
    A base class to represent calculation-volume-boundaries.
    Use the subclasses for practical implementations.
    """
    def __init__(self, grid: Union[Grid, Sequence, jax.Array, None] = None,
                 thickness: Union[float, Sequence, jax.Array] = 0.0,
                 background_permittivity: complex = 1.0):
        """
        :param grid: The Grid to which to the boundaries will be applied.
        :param thickness: The thickness as a scalar, vector, or 2d-array (axes x side). Broadcasting is used as necessary.
        :param background_permittivity: The background permittivity of the boundary (default: 1.0 for vacuum). This is
            only used when the absolute permittivity is requested.
        """
        if not isinstance(grid, Grid):
            grid = Grid.from_ranges(grid)
        self.__grid = grid
        self.__thickness = jnp.broadcast_to(jnp.asarray(thickness), (self.grid.ndim, 2))
        self.__background_permittivity = background_permittivity

    @property
    def grid(self) -> Grid:
        """The Cartesian grid that indicates the sample positions of this bound and the volume it encompasses."""
        return self.__grid

    @property
    def thickness(self) -> jax.Array:
        """
        The thickness as a 2D-array `thickness[axis, front_back]` in meters.
        """
        return self.__thickness.copy()

    @property
    def background_permittivity(self) -> complex:
        """A complex scalar indicating the permittivity of the background."""
        return self.__background_permittivity

    @property
    def electric_susceptibility(self) -> jax.Array:
        """
        The electric susceptibility, chi_E, at every sample point.
        Note that the returned array may have singleton dimensions that must be broadcast!
        """
        return jnp.zeros(self.grid.shape)

    @property
    def magnetic_susceptibility(self) -> jax.Array:
        """
        The magnetic susceptibility, chi_H, at every sample point.
        Note that the returned array may have singleton dimensions that must be broadcast!
        """
        return jnp.zeros(self.grid.shape)

    @property
    def between(self) -> jax.Array:
        """
        Returns a boolean array indicating True for the voxels between the boundaries where the calculation happens.
        The inner edge is considered in between the boundaries.
        """
        result = jnp.asarray(True)
        for axis in range(self.grid.ndim):
            rng = self.grid[axis]
            result = jnp.logical_and(result, jnp.logical_and(rng.ravel()[0] + self.thickness[axis, 0] <= rng, rng <= rng.ravel()[-1] - self.thickness[axis, 1]))
        return result

    @property
    def beyond(self) -> jax.Array:
        """
        Returns a boolean array indicating True for the voxels beyond the boundary edge, i.e. inside the boundaries, where the calculation should be ignored.
        """
        return jnp.logical_not(self.between)


class PeriodicBound(Bound):
    def __init__(self, grid: Union[Grid, Sequence, jax.Array]):
        """
        Constructs an object that represents periodic boundaries.

        :param grid: The Grid to which to the boundaries will be applied.
        """
        super().__init__(grid=grid, thickness=0.0)


class AbsorbingBound(Bound, Electric):
    def __init__(self, grid: Union[Grid, Sequence, jax.Array], thickness: Union[float, Sequence, jax.Array] = 0.0,
                 extinction_coefficient_function: Union[Callable, Sequence, jax.Array] = lambda rel_depth: rel_depth,
                 background_permittivity: complex = 1.0):
        """
        Constructs a boundary with depth-dependent extinction coefficient, kappa(rel_depth).

        :param grid: The Grid to which to the boundaries will be applied.
        :param thickness: The boundary thickness(es) in meters. This can be specified as a 2d-array [axis, side].
            Singleton dimensions are broadcast.
        :param extinction_coefficient_function: A function that returns the extinction coefficient as function of
            the depth in the boundary relative to the total thickness of the boundary.
        :param background_permittivity: (default: 1.0 for vacuum)
        """
        super().__init__(grid=grid, thickness=thickness, background_permittivity=background_permittivity)

        if isinstance(extinction_coefficient_function, Callable):
            extinction_coefficient_function = [extinction_coefficient_function] * 2
        if isinstance(extinction_coefficient_function[0], Callable) and len(extinction_coefficient_function) == 1:
            extinction_coefficient_function = grid.ndim * [extinction_coefficient_function]
        if len(extinction_coefficient_function) == 1:
            extinction_coefficient_function = grid.ndim * extinction_coefficient_function
        extinction_coefficient_function = [2 * _ if len(_) == 1 else _ for _ in extinction_coefficient_function]

        self.__extinction_coefficient_functions = extinction_coefficient_function

    @property
    def is_electric(self) -> bool:
        return True

    @property
    def extinction(self) -> jax.Array:
        """
        Determines the extinction coefficient, kappa, of the boundary on a plaid grid.
        The only non-zero values are found in the boundaries. At the corners, the maximum extinction value of the
        overlapping dimensions is returned.

        Note that the returned array may have singleton dimensions that must be broadcast!

        :return: An nd-array with the extinction coefficient, kappa.
        """
        kappa = 0.0
        for axis, rng in enumerate(self.grid):
            for back_side in range(2):
                thickness = self.thickness[axis, back_side] * jnp.sign(self.grid.step[axis])
                if not back_side:
                    new_depth_in_boundary = (rng.ravel()[0] + thickness) - rng
                else:
                    new_depth_in_boundary = rng - (rng.ravel()[-1] - thickness)
                new_depth_in_boundary *= jnp.sign(self.grid.step[axis])
                in_boundary = new_depth_in_boundary > 0
                if jnp.any(in_boundary):
                    rel_depth = in_boundary * new_depth_in_boundary / thickness
                    kappa_function = self.__extinction_coefficient_functions[axis][back_side]
                    kappa = jnp.maximum(kappa, kappa_function(rel_depth) * in_boundary)
        return kappa

    @property
    def electric_susceptibility(self) -> jax.Array:
        """
        The electric susceptibility, chi_E, at every sample point.
        Note that the returned array may have singleton dimensions that must be broadcast!
        """
        n = jnp.sqrt(self.background_permittivity)
        epsilon = (n + 1j * self.extinction)**2
        return epsilon - self.background_permittivity


class LinearBound(AbsorbingBound):
    def __init__(self, grid: Union[Grid, Sequence, jax.Array], thickness: Union[float, Sequence, jax.Array] = 0.0,
                 max_extinction_coefficient: Union[float, Sequence, jax.Array] = 0.25,
                 background_permittivity: complex = 1.0):
        """
        Constructs a boundary with linearly increasing extinction coefficient, kappa.

        :param grid: The Grid to which to the boundaries will be applied.
        :param thickness: The boundary thickness(es) in meters. This can be specified as a 2d-array [axis, side].
            Singleton dimensions are broadcast.
        :param max_extinction_coefficient: The maximum extinction coefficient, reached at the deepest point of the
            boundary at the edge of the calculation volume.
        :param background_permittivity: (default: 1.0 for vacuum)
        """
        # Define a linear function for every axis and every side
        max_extinction_coefficient = jnp.atleast_2d(max_extinction_coefficient)
        self._max_extinction_coefficient = max_extinction_coefficient

        def linear_gradient(kappa_max):
            if kappa_max.ndim > 0:
                return tuple(map(linear_gradient, kappa_max))
            return lambda _: kappa_max * _

        super().__init__(grid=grid, thickness=thickness,
                         extinction_coefficient_function=linear_gradient(max_extinction_coefficient),
                         background_permittivity=background_permittivity)
