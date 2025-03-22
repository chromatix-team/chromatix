"""
Code ported from
[this file in MacroMax](https://github.com/tttom/MacroMax/blob/master/python/macromax/utils/display/grid2extent.py)
and adapted for JAX.
"""
import warnings

import jax.numpy as jnp

from chromatix.utils.grid import Grid


def grid2extent(*args, origin_lower: bool = False):
    """
    Utility function to determine extent values for matplotlib.pyploy.imshow

    :param args: A Grid object or monotonically increasing ranges, one per dimension (vertical, horizontal)
    :param origin_lower: (default: False) Set this to True when imshow has the origin set to 'lower' to have the
        vertical axis increasing upwards.

    :return: An array with 4 numbers indicating the extent of the displayed data.
    """
    if isinstance(args[0], Grid):
        ranges = args[0]
    else:
        ranges = args
    if len(ranges) > 2:
        warnings.warn(f'Only using the last two axes in grid2extent(...)!')
    extent = []
    for idx, rng in enumerate(ranges[:-3:-1]):
        rng = jnp.array(rng).ravel()
        if len(rng) < 2:
            raise ValueError('The function grid2extent requires a grid with 2 dimensions greater than 1.')
        step = rng[1] - rng[0]
        first, last = rng[0] - 0.5 * step, rng[-1] + 0.5 * step
        if not origin_lower and idx == 1:
            first, last = last, first
        extent.append(first)
        extent.append(last)

    return jnp.asarray(extent)
