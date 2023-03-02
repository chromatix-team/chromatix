import jax.numpy as jnp

from ..field import Field
from einops import rearrange
from chex import Array, assert_rank
from typing import Optional, Sequence, Tuple
from chromatix.utils import create_grid, grid_spatial_to_pupil

__all__ = [
    "phase_change",
    "flat_phase",
    "potato_chip",
    "defocused_ramps",
    "wrap_phase",
]


# Field function
def phase_change(field: Field, phase: Array) -> Field:
    """
    Perturbs ``field`` by ``phase`` (given in radians).

    Returns a new ``Field`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        phase: The phase to apply.
    """
    assert_rank(phase, 4, custom_message="Phase must be array of shape [1 H W 1]")
    return field * jnp.exp(1j * phase)


# Phase mask initializations
def flat_phase(shape: Tuple[int, ...], value: float = 0.0) -> Array:
    """
    Computes a flat phase mask (one with constant value).

    Args:
        shape: The shape of the phase mask, described as a tuple of
            integers of the form (1 H W 1).
        value: The constant value to use for the phase mask, defaults to 0.
    """
    return jnp.full(shape, value)


def potato_chip(
    shape: Tuple[int, ...],
    spacing: float,
    wavelength: float,
    n: float,
    f: float,
    NA: float,
    d: float = 50.0,
    C0: float = -146.7,
) -> Array:
    """
    Computes the "potato chip" phase mask described by [1].

    Also known as the "helical focus" phase mask, this phase mask was designed
    to produce an extended helical PSF for 3D snapshot microscopy.

    [1]: Broxton, Michael. "Volume reconstruction and resolution limits for
        three dimensional snapshot microscopy."
        Dissertation, Stanford University, 2017.

    Args:
        shape: The shape of the phase mask, described as a tuple of
            integers of the form (1 H W 1).
        spacing: The spacing of each pixel in the phase mask.
        wavelength: The wavelength to compute the phase mask for.
        n: Refractive index.
        f: The focal distance (should be in same units as ``wavelength``).
        NA: The numerical aperture. Phase will be 0 outside of this NA.
        d: Sets the axial extent of the PSF (should be in same units as
            ``wavelength``). Defaults to 50 microns, as shown in [1]. See [1]
            for more details.
        C0: Adjusts the focus of the PSF. Set to value described in [1]. See
            [1] for more details.
    """
    # @copypaste(Field): We must use meshgrid instead of mgrid here
    # in order to be jittable
    grid = create_grid(shape, spacing)
    # Normalize coordinates from -1 to 1 within radius R
    grid = grid_spatial_to_pupil(grid, f, NA, n)
    l2_sq_grid = jnp.sum(grid**2, axis=0)
    theta = jnp.arctan2(*grid)
    k = n / wavelength
    phase = theta * (d * jnp.sqrt(k**2 - l2_sq_grid) + C0)
    phase *= l2_sq_grid < 1
    return phase


def defocused_ramps(
    shape: Tuple[int, ...],
    spacing: float,
    wavelength: float,
    n: float,
    f: float,
    NA: float,
    num_ramps: int = 6,
    delta: Sequence[float] = [2374.0] * 6,
    defocus: Sequence[float] = [-50.0, 150.0, -100.0, 50.0, -150.0, 100.0],
) -> Array:
    """
    Computes the "defocused ramps" phase mask as described in [1].

    This phase mask is intended to be used in a 4f microscope to produce a
    number of "pencil" beams in the resulting PSF. The resulting PSF produces
    multiple subimages of the sample on the camera that are projections of the
    sample along different angles and through different axial ranges, intended
    to be used for 3D snapshot microscopy.

    The name describes the fact that the phase mask consists of multiple phase
    ramps around a central flat region, combined with a bowl of defocus within
    each phase ramp to defocus the pencil beam that results from that arm of
    the phase mask.

    [1]: Deb et al. "FourierNets enable the design of highly non-local optical
        encoders for computational imaging." NeurIPS, 2022.

    Args:
        shape: The shape of the phase mask, described as a tuple of
            integers of the form (1 H W 1).
        spacing: The spacing of each pixel in the phase mask.
        wavelength: The wavelength to compute the phase mask for.
        n: Refractive index.
        f: The focal distance (should be in same units as ``wavelength``).
        NA: The numerical aperture. Phase will be 0 outside of this NA.
        num_ramps: Sets the number of "pencil" beams or "ramps". The number of
            pencil beams will be ``num_ramps + 1``, because of the central
            flat region of the phase mask.
        delta: Controls the "slope" of each phase ramp. Higher values move the
            resulting pencil further away from the center of the field.
        defocus: Controls the defocus of each pencil axially (should be in
            same units as ``wavelength``).
    """
    grid = create_grid(shape, spacing)
    # Normalize coordinates from -1 to 1 within radius R
    grid = grid_spatial_to_pupil(grid, f, NA, n)
    l2_sq_grid = jnp.sum(grid**2, axis=0)
    theta = jnp.arctan2(*grid)
    edges = jnp.linspace(-jnp.pi, jnp.pi, num_ramps + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    flat_region_edge = (num_ramps + 1) ** -0.5
    defocus_center = (flat_region_edge + 1) / 2.0
    phase = jnp.zeros(shape)

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
    """
    Wraps values of ``phase`` to the range given by ``limits``.

    Args:
        phase: The phase mask to wrap (in radians).
        limits: A tuple defining the minimum and maximum value that ``phase``
            will be wrapped to.
    """
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


def spectrally_modulate_phase(
    phase: Array, spectrum: Array, central_wavelength: float
) -> Array:
    """Spectrally modulates a given ``phase`` for multiple wavelengths."""
    assert_rank(spectrum, 4, custom_message="Spectrum must be array of shape [1 1 1 C]")

    spectral_modulation = central_wavelength / spectrum
    return phase * spectral_modulation
