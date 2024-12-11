import jax.numpy as jnp
import pytest
from chromatix.utils import (
    create_grid,
    grid_spatial_to_pupil,
    zernike_aberrations,
)


def test_first_ten_zernikes():
    size = (256, 256)
    spacing = 0.3
    f = 100.0
    n = 1.33
    NA = 0.5
    wavelength = 0.532

    # @copypaste(Field): We must use meshgrid instead of mgrid here
    # in order to be jittable
    grid = create_grid(size, spacing)
    # Normalize coordinates from -1 to 1 within radius R
    grid = grid_spatial_to_pupil(grid, f, NA, n)
    rho = jnp.sqrt(jnp.sum(grid**2, axis=0))  # radial coordinate
    mask = rho <= 1
    radius = rho * mask
    angle = jnp.arctan2(*grid) * mask  # angle coordinate

    ansi_indices = list(range(10))

    expected = [
        piston(mask),
        y_tilt(mask, radius, angle),
        x_tilt(mask, radius, angle),
        oblique_astigmatism(mask, radius, angle),
        defocus(mask, radius),
        vertical_astigmatism(mask, radius, angle),
        vertical_trefoil(mask, radius, angle),
        vertical_coma(mask, radius, angle),
        horizontal_coma(mask, radius, angle),
        oblique_trefoil(mask, radius, angle),
    ]

    print(ansi_indices)
    for idx in ansi_indices:
        print(f"Testing Zernike polynomial {idx}.")
        phase = zernike_aberrations(
            size,
            spacing,
            wavelength,
            n,
            f,
            NA,
            ansi_indices=[idx],
            coefficients=[1],
        )
        assert phase.shape == size
        assert jnp.allclose(
            phase.squeeze(), 2 * jnp.pi * expected[idx] / wavelength
        ), f"Mismatch in Zernike polynomial {idx}."


def piston(mask):
    return jnp.ones_like(mask) * mask


def y_tilt(mask, r, theta):
    return 2 * r * jnp.sin(theta) * mask


def x_tilt(mask, r, theta):
    return 2 * r * jnp.cos(theta) * mask


def oblique_astigmatism(mask, r, theta):
    return jnp.sqrt(6) * r**2 * jnp.sin(2 * theta) * mask


def defocus(mask, r):
    return jnp.sqrt(3) * (2 * r**2 - 1) * mask


def vertical_astigmatism(mask, r, theta):
    return jnp.sqrt(6) * r**2 * jnp.cos(2 * theta) * mask


def vertical_trefoil(mask, r, theta):
    return jnp.sqrt(8) * r**3 * jnp.sin(3 * theta) * mask


def vertical_coma(mask, r, theta):
    return jnp.sqrt(8) * (3 * r**3 - 2 * r) * jnp.sin(theta) * mask


def horizontal_coma(mask, r, theta):
    return jnp.sqrt(8) * (3 * r**3 - 2 * r) * jnp.cos(theta) * mask


def oblique_trefoil(mask, r, theta):
    return jnp.sqrt(8) * r**3 * jnp.cos(3 * theta) * mask


def primary_spherical(mask, r, theta):
    return jnp.sqrt(5) * (6 * r**4 - 6 * r**2 + 1) * mask


if __name__ == "__main__":
    pytest.main([__file__])
