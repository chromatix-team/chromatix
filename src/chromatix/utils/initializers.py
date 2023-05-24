import jax.numpy as jnp
from einops import rearrange
from chex import Array
from typing import Sequence, Tuple
from .utils import create_grid, grid_spatial_to_pupil
from scipy.special import comb  # type: ignore
import math

__all__ = [
    "flat_phase",
    "potato_chip",
    "seidel_aberrations",
    "zernike_aberrations",
    "defocused_ramps",
]


def flat_phase(shape: Tuple[int, int], *args, value: float = 0.0) -> Array:
    """
    Computes a flat mask (one with constant value).

    Args:
        shape: The shape of the mask, described as a tuple of
            integers of the form (H W).
        value: The constant value to use for the mask, defaults to 0.
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
            integers of the form (H W).
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


def seidel_aberrations(
    shape: Tuple[int, int],
    spacing: float,
    wavelength: float,
    n: float,
    f: float,
    NA: float,
    coefficients: Sequence[float],
    u: float = 0,
    v: float = 0,
) -> Array:
    """
    Computes the Seidel phase polynomial described by [1]. Accounts for spatially
    varying aberrations by creating a different phase mask for each object field
    position (u,v)

    [1]: Voelz, David George. Computational fourier optics: a MATLAB tutorial. Vol. 534.
    Bellingham, Washington: SPIE press, 2011.

    Args:
        shape: The shape of the phase mask, described as a tuple of
            integers of the form (H W).
        spacing: The spacing of each pixel in the phase mask.
        wavelength: The wavelength to compute the phase mask for.
        n: Refractive index.
        f: The focal distance (should be in same units as ``wavelength``).
        NA: The numerical aperture. Phase will be 0 outside of this NA.
        coefficients: weight coefficients for Seidel aberrations
        u: The horizontal position of the object field point
        v: The vertical position of the object field point
    """
    # @copypaste(Field): We must use meshgrid instead of mgrid here
    # in order to be jittable
    grid = create_grid(shape, spacing)
    # Normalize coordinates from -1 to 1 within radius R
    grid = grid_spatial_to_pupil(grid, f, NA, n)
    Y, X = grid

    rot_angle = jnp.arctan2(v, u)

    obj_rad = jnp.sqrt(u**2 + v**2)

    X_rot = X * jnp.cos(rot_angle) + Y * jnp.sin(rot_angle)
    Y_rot = -X * jnp.sin(rot_angle) + Y * jnp.cos(rot_angle)

    pupil_radii = jnp.square(X_rot) + jnp.square(Y_rot)
    phase = (
        wavelength * coefficients[0] * jnp.square(pupil_radii)
        + wavelength * coefficients[1] * obj_rad * pupil_radii * X_rot
        + wavelength * coefficients[2] * (obj_rad**2) * jnp.square(X_rot)
        + wavelength * coefficients[3] * (obj_rad**2) * pupil_radii
        + wavelength * coefficients[4] * (obj_rad**3) * X_rot
    )

    l2_sq_grid = X**2 + Y**2

    phase *= l2_sq_grid < 1
    return phase


def zernike_aberrations(
    shape: Tuple[int, int],
    spacing: float,
    wavelength: float,
    n: float,
    f: float,
    NA: float,
    ansi_indices: Sequence[int],
    coefficients: Sequence[float],
) -> Array:
    """
    Computes Zernike aberrations

    Args:
        shape: The shape of the phase mask, described as a tuple of
            integers of the form (H W).
        spacing: The spacing of each pixel in the phase mask.
        wavelength: The wavelength to compute the phase mask for.
        n: Refractive index.
        f: The focal distance (should be in same units as ``wavelength``).
        NA: The numerical aperture. Phase will be 0 outside of this NA.
        ansi_indices: linear Zernike indices according to ANSI numbering
        coefficients: weight coefficients for the Zernike polynomials
    """

    def convert_ansi_to_zernike_indices(indices):
        d = [math.sqrt(9 + 8 * ind) for ind in indices]
        n = [math.ceil((x - 3) / 2) for x in d]
        m = [2 * ind - x * (x + 2) for ind, x in zip(indices, n)]
        return tuple(zip(n, m))

    def radial_polynomial(n, m):
        """Returns a function calculating the specified radial polynomial."""

        def R(rho):
            if (n - m) % 2 == 0:
                sum = 0
                for k in range(int((n - m) / 2) + 1):
                    sum += (
                        rho ** (n - 2 * k)
                        * ((-1) ** k)
                        * comb(n - k, k)
                        * comb(n - 2 * k, (n - m) / 2 - k)
                    )
                R_nm = sum
            else:
                R_nm = 0
            return R_nm

        return R

    # @copypaste(Field): We must use meshgrid instead of mgrid here
    # in order to be jittable
    grid = create_grid(shape, spacing)
    # Normalize coordinates from -1 to 1 within radius R
    grid = grid_spatial_to_pupil(grid, f, NA, n)

    rho = jnp.sum(grid**2, axis=0)  # radial coordinate

    mask = rho <= 1
    rho = rho * mask
    theta = jnp.arctan2(*grid) * mask  # angle coordinate

    # construct zernike bases to combine during forward pass
    zernike_polynomials = []
    zernike_indices = convert_ansi_to_zernike_indices(ansi_indices)

    for n, m in zernike_indices:
        calc_polynomial = radial_polynomial(n, m)
        R_nm = calc_polynomial(rho)

        if m == 0:
            Z = R_nm
        elif m > 0:  # 'even' Zernike polynomials
            Z = R_nm * jnp.cos(theta * abs(m))
        else:  # 'odd' Zernike polynomials
            Z = R_nm * jnp.sin(theta * abs(m))

        Z = Z * mask
        zernike_polynomials.append(Z)

    zernike_polynomials = jnp.asarray(zernike_polynomials)
    zernike_polynomials = rearrange(zernike_polynomials, "b h w -> h w b")

    phase = jnp.dot(zernike_polynomials, jnp.asarray(coefficients))

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
            integers of the form (H W).
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
