import jax.numpy as jnp

from ..field import Field
from typing import Optional
from .pupils import circular_pupil
from ..ops.fft import optical_fft

__all__ = ["thin_lens", "ff_lens", "df_lens"]


def thin_lens(field: Field, f: float, n: float, NA: Optional[float] = None) -> Field:
    """_summary_

    Args:
        field (Field): _description_
        f (float): _description_
        n (float): _description_
        NA (Optional[float], optional): _description_. Defaults to None.

    Returns:
        Field: _description_
    """
    L = jnp.sqrt(field.spectrum * f / n)
    phase = -jnp.pi * field.l2_sq_grid / L**2

    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    return field * jnp.exp(1j * phase)


def ff_lens(
    field: Field,
    f: float,
    n: float,
    NA: Optional[float] = None,
    inverse: bool = False,
    loop_axis: Optional[int] = None,
) -> Field:
    """_summary_

    Args:
        field (Field): _description_
        f (float): _description_
        n (float): _description_
        NA (Optional[float], optional): _description_. Defaults to None.
        inverse (bool, optional): _description_. Defaults to False.
        loop_axis (Optional[int], optional): _description_. Defaults to None.

    Returns:
        Field: _description_
    """
    # Pupil
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    return optical_fft(field, f, n, loop_axis, inverse)


def df_lens(
    field: Field,
    d: float,
    f: float,
    n: float,
    NA: Optional[float] = None,
    inverse: bool = False,
    loop_axis: Optional[int] = None,
) -> Field:
    """_summary_

    Args:
        field (Field): _description_
        d (float): _description_
        f (float): _description_
        n (float): _description_
        NA (Optional[float], optional): _description_. Defaults to None.
        inverse (bool, optional): _description_. Defaults to False.
        loop_axis (Optional[int], optional): _description_. Defaults to None.

    Returns:
        Field: _description_
    """
    # Preliminaries
    L = jnp.sqrt(field.spectrum * f / n)  # Lengthscale L

    # Phase factor due to distance d from lens
    phase = jnp.pi * (1 - d / f) * field.l2_sq_grid / L**2

    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    return optical_fft(field, f, n, loop_axis, inverse) * jnp.exp(1j * phase)
