import jax.numpy as jnp

from ..field import Field
from chex import Array, assert_rank
from typing import Optional, Sequence, Tuple

__all__ = [
    "phase_change",
    "flat_phase",
    "potato_chip",
    "defocused_ramps",
    "wrap_phase",
]


# Field function
def phase_change(field: Field, phase: Array) -> Field:
    assert_rank(phase, 4, custom_message="Phase must be array of shape [1 H W 1]")
    return field * jnp.exp(1j * phase)


# Phase mask initializations
def flat_phase(shape: Tuple[int, int], value: float = 0.0) -> Array:
    return jnp.full((1, shape[0], shape[1], 1), value)


def potato_chip(
    shape: Tuple[int, int],
    spacing: float,
    wavelength: float,
    n: float,
    f: float,
    NA: float,
    d: float = 50.0,
    C0: float = -146.7,
) -> Array:
    # @copypaste(Field): We must use meshgrid instead of mgrid here
    # in order to be jittable
    half_size = jnp.array(shape) / 2
    grid = jnp.meshgrid(
        jnp.linspace(-half_size[0], half_size[0] - 1, num=shape[0]) + 0.5,
        jnp.linspace(-half_size[1], half_size[1] - 1, num=shape[1]) + 0.5,
        indexing="ij",
    )
    grid = spacing * rearrange(grid, "d h w -> d 1 h w 1")
    # Normalize coordinates from -1 to 1 within radius R
    R = (wavelength * f) / n
    grid = (grid / R) / (NA / wavelength)
    l2_sq_grid = jnp.sum(grid**2, axis=0)
    theta = jnp.arctan2(*grid)
    k = n / wavelength
    phase = theta * (d * jnp.sqrt(k**2 - l2_sq_grid) + C0)
    phase *= l2_sq_grid < 1
    return phase


def defocused_ramps(
    shape: Tuple[int, int],
    spacing: float,
    wavelength: float,
    n: float,
    f: float,
    NA: float,
    num_ramps: int = 6,
    delta: Sequence[float] = [2374.0] * 6,
    defocus: Sequence[float] = [-50.0, 150.0, -100.0, 50.0, -150.0, 100.0],
) -> Array:
    # @copypaste(Field): We must use meshgrid instead of mgrid here
    # in order to be jittable
    half_size = jnp.array(shape) / 2
    grid = jnp.meshgrid(
        jnp.linspace(-half_size[0], half_size[0] - 1, num=shape[0]) + 0.5,
        jnp.linspace(-half_size[1], half_size[1] - 1, num=shape[1]) + 0.5,
        indexing="ij",
    )
    grid = spacing * rearrange(grid, "d h w -> d 1 h w 1")
    # Normalize coordinates from -1 to 1 within radius R
    R = (wavelength * f) / n
    grid = (grid / R) / (NA / wavelength)
    l2_sq_grid = jnp.sum(grid**2, axis=0)
    theta = jnp.arctan2(*grid)
    edges = jnp.linspace(-jnp.pi, jnp.pi, num_ramps + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    flat_region_edge = (num_ramps + 1) ** -0.5
    defocus_center = (flat_region_edge + 1) / 2.0
    phase = jnp.zeros((1, shape[0], shape[1], 1))

    def ramp(center, theta_bounds, delta_ramp, ramp_defocus):
        # Calculate distances along and across current ramp
        ramp_parallel_distance = grid[0] * jnp.sin(center) + grid[1] * jnp.cos(center)
        ramp_perpendicular_distance = grid[0] * jnp.cos(center) - grid[1] * jnp.sin(
            center
        )
        # Select coordinates for current ramp
        ramp_mask = (
            (theta >= theta_bounds[0])
            & (theta < theta_bounds[1])
            & (ramp_parallel_distance > flat_region_edge)
            & (l2_sq_grid < 1)
        )
        # Create ramp
        phase = ramp_mask * delta_ramp * ramp_perpendicular_distance
        # Create defocus within ramp
        ramp_quadratic = (grid[1] - jnp.cos(center) * defocus_center) ** 2 + (
            grid[0] - jnp.sin(center) * defocus_center
        ) ** 2
        phase += ramp_mask * (ramp_defocus * ramp_quadratic)
        phase -= ramp_mask * jnp.where(ramp_mask > 0, phase, 0).mean()
        return phase

    for ramp_idx in range(num_ramps):
        phase += ramp(
            centers[ramp_idx],
            edges[ramp_idx : (ramp_idx + 2)],
            delta[ramp_idx],
            defocus[ramp_idx],
        )
    phase *= l2_sq_grid < 1
    return phase


# Utility functions
def wrap_phase(phase: Array, limits: Tuple[float, float] = (-jnp.pi, jnp.pi)) -> Array:
    phase_min, phase_max = limits
    assert phase_min < phase_max, "Lower limit needs to be smaller than upper limit."
    min_indices = phase < phase_min
    max_indices = phase > phase_max
    phase = phase.at[min_indices].set(
        phase[min_indices]
        + 2 * jnp.pi * (1 + (phase_min - phase[min_indices]) // (2 * jnp.pi))
    )
    phase = phase.at[max_indices].set(
        phase[max_indices]
        - 2 * jnp.pi * (1 + (phase[max_indices] - phase_max) // (2 * jnp.pi))
    )
    return phase
