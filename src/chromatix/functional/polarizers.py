from ..field import VectorField
from typing import Union
from chex import Array
import jax.numpy as jnp

__all__ = [
    # General functions
    "jones_vector",
    "polarizer",
    "phase_retarder",
    # Initialisers
    "linear",
    "left_circular",
    "right_circular",
    # Polarisers
    "linear_polarizer",
    "left_circular_polarizer",
    "right_circular_polarizer",
    # Waveplates
    "wave_plate",
    "halfwave_plate",
    "quarterwave_plate",
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


def left_circular() -> Array:
    """Generates a jones vector for circularly polarised light."""
    return jones_vector(jnp.pi / 4, jnp.pi / 2)


def right_circular() -> Array:
    """Generates a jones vector for circularly polarised light."""
    return jones_vector(jnp.pi / 4, -jnp.pi / 2)


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


# ================== Wave plates =======================
def phase_retarder(
    field: VectorField, theta: float, eta: float, phi: float
) -> VectorField:
    s, c = jnp.sin(theta), jnp.cos(theta)
    scale = jnp.exp(-1j * eta / 2)
    J00 = scale * (c**2 + jnp.exp(1j * eta) * s**2)
    J11 = scale * (s**2 + jnp.exp(1j * eta) * c**2)
    J01 = scale * (1 - jnp.exp(1j * eta)) * jnp.exp(-1j * phi) * s * c
    J10 = scale * (1 - jnp.exp(1j * eta)) * jnp.exp(1j * phi) * s * c
    return polarizer(field, J00, J01, J10, J11)


def wave_plate(field: VectorField, theta: float, eta: float) -> VectorField:
    return phase_retarder(field, theta, eta, phi=0)


def halfwave_plate(field: VectorField, theta: float) -> VectorField:
    return phase_retarder(field, theta, eta=jnp.pi, phi=0)


def quarterwave_plate(field: VectorField, theta: float) -> VectorField:
    return phase_retarder(field, theta, eta=jnp.pi / 2, phi=0)
