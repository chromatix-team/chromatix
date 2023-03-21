import jax.numpy as jnp
from typing import Union, Tuple
from ..field import Field


def pad(field: Field, pad_width: Union[int, Tuple[int, int]], cval: float = 0) -> Field:
    if isinstance(pad_width, int):
        pad_width = (pad_width, pad_width)
    u = jnp.pad(
        field.u,
        [(n, n) for n in (0,) * (field.ndim - 3) + (*pad_width, 0)],
        constant_values=cval,
    )
    return field.replace(u=u)


def crop(field: Field, crop_width: Union[int, Tuple[int, int]]) -> Field:
    if isinstance(crop_width, int):
        crop_width = (crop_width, crop_width)
    crop = [
        slice(n, size - n)
        for size, n in zip(field.shape, (0,) * (field.ndim - 3) + (*crop_width, 0))
    ]
    return field.replace(u=field.u[tuple(crop)])
