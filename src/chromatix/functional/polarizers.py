import jax.numpy as jnp

from ..field import Field, VectorField
from einops import rearrange

__all__ = ["linear_polarizer", "left_circular_polarizer", "right_circular_polarizer"]


def field_after_polarizer(field: VectorField, J00, J01, J10, J11) -> VectorField:
    LP = jnp.array([[J00, J01, 0], [J10, J11, 0], [0, 0, 0]])
    LP = LP[::-1, ::-1]  # why the inverse?
    return field.replace(u=jnp.dot(field.u, LP.T))


def linear_polarizer(field: VectorField, polarizer_angle: float) -> VectorField:
    """
    Applies a thin polarizer placed directly after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the polarizer will be applied.
        p_angle: linear polarizers oriented to pass light polarized at polarizer_angle

    Returns:
        The ``Field`` directly after the polarizer.
    """
    c = jnp.cos(polarizer_angle)
    s = jnp.sin(polarizer_angle)
    J00 = c**2
    J11 = s**2
    J01 = s * c
    J10 = J01
    return field_after_polarizer(field, J00, J01, J10, J11)


def left_circular_polarizer(field: VectorField) -> VectorField:
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


def right_circular_polarizer(field: VectorField) -> VectorField:
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
