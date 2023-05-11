import jax.numpy as jnp
from chex import Array, assert_rank
from ..field import Field
from ..utils.shapes import _broadcast_2d_to_spatial

__all__ = ["amplitude_change"]


def amplitude_change(field: Field, amplitude: Array) -> Field:
    """
    Perturbs ``field`` by ``amplitude``.

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        amplitude: The amplitude to apply.
    """
    amplitude = _broadcast_2d_to_spatial(amplitude, field.ndim)
    assert_rank(
        amplitude,
        field.ndim,
        custom_message="Amplitude must have same ndim as incoming ``Field``.",
    )
    return field * amplitude.astype(jnp.complex64)
