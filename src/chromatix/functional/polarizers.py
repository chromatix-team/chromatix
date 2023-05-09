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
    # Polarizers
    "linear_polarizer",
    "left_circular_polarizer",
    "right_circular_polarizer",
    # Waveplates
    "wave_plate",
    "halfwave_plate",
    "quarterwave_plate",
]


def jones_vector(theta: float, beta: float) -> Array:
    """Generates a Jones vector given by [cos(theta), sin(theta)exp(1j*beta)].

    Args:
        theta (float): Polarization angle.
        beta (float): Relative delay between components.

    Returns:
        Array: Jones vector.
    """

    # Generates a Jones vector with a given beta = alpha_y - alpha_x.
    # Assumes alpha_x=0.
    return jnp.array(
        [0, jnp.sin(theta) * jnp.exp(1j * beta), jnp.cos(theta)], dtype=jnp.complex64
    )


def linear(theta: float) -> Array:
    """Generates a Jones vector for linearly polarized
    light with an angle $\theta$ w.r.t. to the horizontal.

    Args:
        theta (float): Angle w.r.t horizontal.

    Returns:
        Array: Linearly polarized Jones vector.
    """
    return jones_vector(theta, 0)


def left_circular() -> Array:
    """Generates a Jones vector for left circularly polarized
    light.

    Returns:
        Array: Left circularly polarized Jones vector.
    """
    return jones_vector(jnp.pi / 4, jnp.pi / 2)


def right_circular() -> Array:
    """Generates a Jones vector for right circularly polarized
    light.

    Returns:
        Array: Right circularly polarized Jones vector.
    """
    return jones_vector(jnp.pi / 4, -jnp.pi / 2)


def polarizer(
    field: VectorField,
    J00: Union[float, complex, Array],
    J01: Union[float, complex, Array],
    J10: Union[float, complex, Array],
    J11: Union[float, complex, Array],
) -> VectorField:
    """Applies a Jones matrix with given components to the field.
    Note that the components here refer to the common choice of coordinate
    system and are inverted by us - i.e. J00 refers to Jxx.

    Args:
        field (VectorField): field to apply polarization to.
        J00 (Union[float, complex, Array]): _description_
        J01 (Union[float, complex, Array]): _description_
        J10 (Union[float, complex, Array]): _description_
        J11 (Union[float, complex, Array]): _description_

    Returns:
        VectorField: Field after polarizer.
    """
    # Invert the axes as our order is zyx
    LP = jnp.array([[0, 0, 0], [0, J11, J10], [0, J01, J00]])
    return field.replace(u=jnp.dot(field.u, LP))


def linear_polarizer(field: VectorField, angle: float) -> VectorField:
    """Applies a linear polarizer with a given angle to the incoming field.

    Args:
        field (VectorField): incoming field.
        angle (float): angle w.r.t to the horizontal.

    Returns:
        VectorField: outgoing field.
    """

    c, s = jnp.cos(angle), jnp.sin(angle)
    J00 = c**2
    J11 = s**2
    J01 = s * c
    J10 = J01
    return polarizer(field, J00, J01, J10, J11)


def left_circular_polarizer(field: VectorField) -> VectorField:
    """Applies a left circular polarizer to the incoming field.

    Args:
        field (VectorField): incoming field.

    Returns:
        VectorField: outgoing field.
    """
    J00 = 1
    J11 = 1
    J01 = -1j
    J10 = 1j
    return polarizer(field, J00, J01, J10, J11)


def right_circular_polarizer(field: VectorField) -> VectorField:
    """
    Applies a thin RCP polarizer to the incoming ``Field``.

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


def phase_retarder(
    field: VectorField, theta: float, eta: float, phi: float
) -> VectorField:
    """Applies a general purpose retardation matrix with angle w.r.t horizontal theta,
    relative phase change eta and circularity phi.

    Args:
        field (VectorField): incoming field.
        theta (float): angle w.r.t horizonal axis.
        eta (float): relative phase retardation.
        phi (float): circularity.

    Returns:
        VectorField: outgoing field.
    """
    s, c = jnp.sin(theta), jnp.cos(theta)
    scale = jnp.exp(-1j * eta / 2)
    J00 = scale * (c**2 + jnp.exp(1j * eta) * s**2)
    J11 = scale * (s**2 + jnp.exp(1j * eta) * c**2)
    J01 = scale * (1 - jnp.exp(1j * eta)) * jnp.exp(-1j * phi) * s * c
    J10 = scale * (1 - jnp.exp(1j * eta)) * jnp.exp(1j * phi) * s * c
    return polarizer(field, J00, J01, J10, J11)


def wave_plate(field: VectorField, theta: float, eta: float) -> VectorField:
    """Applies a general waveplate with angle theta and delay eta to the field.

    Args:
        field (VectorField): incoming field.
        theta (float): angle w.r.t horizontal.
        eta (float): relative delay between components.

    Returns:
        VectorField: outgoing field.
    """
    return phase_retarder(field, theta, eta, phi=0)


def halfwave_plate(field: VectorField, theta: float) -> VectorField:
    """Applies a halfwave plate with angle theta to the incoming field.

    Args:
        field (VectorField): incoming field.
        theta (float): angle w.r.t. horizontal.

    Returns:
        VectorField: outgoing field.
    """
    return phase_retarder(field, theta, eta=jnp.pi, phi=0)


def quarterwave_plate(field: VectorField, theta: float) -> VectorField:
    """Applies a quarterwave plate with angle theta to the incoming field.

    Args:
        field (VectorField): incoming field.
        theta (float): angle w.r.t. horizontal.

    Returns:
        VectorField: outgoing field.
    """
    return phase_retarder(field, theta, eta=jnp.pi / 2, phi=0)
