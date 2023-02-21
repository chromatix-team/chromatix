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
    assert_rank(phase, 4, custom_message="Phase must be array of shape [B, H, W, C]")
    return field * jnp.exp(1j * phase)


# Phase inits
def flat_phase(field: Field, value: float = 0.0) -> Array:
    return jnp.full((1, field.shape[1], field.shape[2], 1), value)


def potato_chip(field: Field, d: float, C0: float, n: float, f: float) -> Array:
    theta = jnp.arctan2(*field.grid)
    k = n / field.spectrum[..., 0]
    L = jnp.sqrt(field.spectrum[..., 0] * f)
    phase = theta * (d * jnp.sqrt(k**2 - field.l2_sq_grid / L**2) + C0)
    return phase


def defocused_ramps(
    field: Field,
    D: float,
    num_ramps: int = 6,
    delta: Sequence[float] = [2374.0] * 6,
    defocus: Sequence[float] = [-50.0, 150.0, -100.0, 50.0, -150.0, 100.0],
) -> jnp.ndarray:
    # normalize coordinates from -1 to 1 within radius D
    grid = field.grid / D
    sq_dist = jnp.sum(grid**2, axis=0)
    theta = jnp.arctan2(grid[0], grid[1])

    edges = jnp.linspace(-jnp.pi, jnp.pi, num_ramps + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    flat_region_edge = (num_ramps + 1) ** -0.5
    defocus_center = (flat_region_edge + 1) / 2.0
    phase = jnp.zeros((1, field.shape[1], field.shape[2], 1))

    def ramp(center, theta_bounds, delta_ramp, ramp_defocus):
        # calculate distances along and across current ramp
        ramp_parallel_distance = grid[0] * jnp.sin(center) + grid[1] * jnp.cos(center)
        ramp_perpendicular_distance = grid[0] * jnp.cos(center) - grid[1] * jnp.sin(
            center
        )
        # select coordinates for current ramp
        ramp_mask = (
            (theta >= theta_bounds[0])
            & (theta < theta_bounds[1])
            & (ramp_parallel_distance > flat_region_edge)
            & (sq_dist < 1)
        )
        # create ramp
        phase = ramp_mask * delta_ramp * ramp_perpendicular_distance
        # create defocus within ramp
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
    phase *= sq_dist < 1
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
