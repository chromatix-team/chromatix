from ..field import VectorField
from typing import Union
from chex import Array
import jax.numpy as jnp

__all__ = [
    "polarizer",
    "linear_polarizer",
    "left_circular_polarizer",
    "right_circular_polarizer",
    "jones_vector",
    "linear",
    "circular",
    "linear_horizontal",
    "linear_vertical",
    "left_circular",
    "right_circular",
]

# ===================== Initializers for amplitudes ================================


def jones_vector(theta: float, beta: float) -> Array:
    """Generates a jones vector with a given beta = alpha_y - alpha_x.
    Assumes alpha_x=0."""
    return jnp.array(
        [0, jnp.sin(theta) * jnp.exp(1j * beta), jnp.cos(theta)], dtype=jnp.complex64
    )


def linear(theta: float) -> Array:
    """Generates a jones vector for linearly polarised light."""
    return jones_vector(theta, 0)


def circular(beta: float) -> Array:
    """Generates a jones vector for circularly polarised light."""
    return jones_vector(jnp.pi / 4, beta)


linear_horizontal = linear(0)
linear_vertical = linear(jnp.pi / 2)
left_circular = circular(jnp.pi / 2)
right_circular = circular(-jnp.pi / 2)

# ===================== Jones Polarisers ================================


def polarizer(
    field: VectorField,
    J00: Union[float, complex, Array],
    J01: Union[float, complex, Array],
    J10: Union[float, complex, Array],
    J11: Union[float, complex, Array],
) -> VectorField:
    """Apply polarisation to incoming field defined by Jones coefficients."""
    # Invert the axes as our order is zyx
    LP = jnp.array([[0, 0, 0], [0, J11, J10], [0, J01, J00]])
    return field.replace(u=jnp.dot(field.u, LP))


def linear_polarizer(field: VectorField, angle: float) -> VectorField:
    """
    Applies a thin polarizer with of a given angle w.r.t. the horizontal to the
    incoming ``VectorField``.

    Args:
        field: The ``VectorField`` to which the polarizer will be applied.
        angle: polarizer_angle w.r.t horizontal.

    Returns:
        The ``VectorField`` directly after the polarizer.
    """
    c, s = jnp.cos(angle), jnp.sin(angle)
    J00 = c**2
    J11 = s**2
    J01 = s * c
    J10 = J01
    return polarizer(field, J00, J01, J10, J11)


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
    return polarizer(field, J00, J01, J10, J11)


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
    return polarizer(field, J00, J01, J10, J11)
