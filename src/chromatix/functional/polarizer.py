import jax.numpy as jnp

from ..field import Field
from einops import rearrange

__all__ = ["linear_polarizer"]


def field_after_polarizer(field: Field, J00, J01, J10, J11):
    LP = jnp.array([[J00, J01, 0], [J10, J11, 0], [0, 0, 0]])
    LP = rearrange(LP[::-1, ::-1], "a b -> 1 a b 1 1 1")
    u = jnp.einsum("ijjklm, ijklm -> ijklm", LP, field.u)
    return field.replace(u=u)


def linear_polarizer(field: Field, p_angle: float) -> Field:
    """
    Applies a thin polarizer placed directly after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the polarizer will be applied.
        p_angle: linear polarizers oriented to pass light polarized at p_angle

    Returns:
        The ``Field`` directly after the polarizer.
    """
    c = jnp.cos(p_angle)
    s = jnp.sin(p_angle)
    J00 = c**2
    J11 = s**2
    J01 = s * c
    J10 = J01
    return field_after_polarizer(field, J00, J01, J10, J11)


def left_circular_polarizer(field: Field) -> Field:
    """
    Applies a thin LCP linear polarizer placed directly after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the polarizer will be applied.

    Returns:
        The ``Field`` directly after the polarizer.
    """
    J00 = 1
    J11 = 1
    J01 = -1j
    J10 = 1j
    return field_after_polarizer(field, J00, J01, J10, J11)


def right_circular_polarizer(field: Field) -> Field:
    """
    Applies a thin RCP polarizer placed directly after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the polarizer will be applied.

    Returns:
        The ``Field`` directly after the polarizer.
    """
    J00 = 1
    J11 = 1
    J01 = 1j
    J10 = -1j
    return field_after_polarizer(field, J00, J01, J10, J11)
