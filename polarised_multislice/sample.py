import jax.numpy as jnp
import numpy as np
from einops import reduce
from jax import Array
from jax.typing import ArrayLike


def Rx(theta: ArrayLike) -> Array:
    """
    Rotation matrix around x-axis for z-y-x vector order.
    :param theta: Angle of rotation in radians
    :return: 3x3 rotation matrix
    """
    return jnp.array(
        [
            [jnp.cos(theta), jnp.sin(theta), 0],
            [-jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def Ry(theta: ArrayLike) -> Array:
    """
    Rotation matrix around y-axis for z-y-x vector order.
    :param theta: Angle of rotation in radians
    :return: 3x3 rotation matrix
    """
    return jnp.array(
        [
            [jnp.cos(theta), 0, -jnp.sin(theta)],
            [0, 1, 0],
            [jnp.sin(theta), 0, jnp.cos(theta)],
        ]
    )


def Rz(theta: ArrayLike) -> Array:
    """
    Rotation matrix around z-axis for z-y-x vector order.
    :param theta: Angle of rotation in radians
    :return: 3x3 rotation matrix
    """
    return jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(theta), jnp.sin(theta)],
            [0, -jnp.sin(theta), jnp.cos(theta)],
        ]
    )


def R(theta_z: ArrayLike, theta_y: ArrayLike, theta_x: ArrayLike) -> Array:
    return Rx(theta_x) @ Ry(theta_y) @ Rz(theta_z)


def single_bead_sample(
    n_background: float,
    n_bead: ArrayLike,
    orientation: ArrayLike,
    radius: float,
    shape: tuple,
    spacing: float,
    k0: float,
    antialiasing: int = 1 
) -> Array:
    """
    Generate a single bead sample in a background medium.

    This function creates a 3D sample containing a single bead with specified
    properties in a background medium. The bead is represented by a spherical
    mask within the sample volume.

    Parameters
    ----------
    n_background : float
        Refractive index of the background medium.
    n_bead : ArrayLike
        Array-like object of length 3 representing the refractive indices
        of the bead along its principal axes.
    orientation : ArrayLike
        Array-like object of length 3 representing the rotation angles
        (in radians) for orienting the bead. The order of rotations is
        assumed to be Z-Y-X (extrinsic rotations).
    radius : float
        Radius of the bead.
    shape : tuple
        3-tuple specifying the shape of the sample volume (z, y, x).
    spacing : float
        Grid spacing in the sample volume.
    k0 : float
        Wavenumber in vacuum.

    Returns
    -------
    Array
        4D array of shape (*shape, 3, 3) representing the sample. Each point
        in the 3D volume contains a 3x3 matrix representing the difference
        between the permittivity tensor of the bead and the background.


    """
    # Making grid and mask
    spacing = spacing / antialiasing
    grid = jnp.mgrid[: antialiasing * (shape[0] + 1), : antialiasing * (shape[1] +1) , : antialiasing * (shape[2] + 1)]
    grid = grid - jnp.mean(grid, axis=(1, 2, 3), keepdims=True)
    mask = jnp.sum(grid[..., None, None] ** 2, axis=0) < (radius / spacing) ** 2

    # Making bead and background
    bead_permitivitty = R(*orientation).T @ jnp.diag(n_bead**2) @ R(*orientation)
    background_permitivitty = jnp.eye(3) * n_background**2

    # Making sample
    permitivitty = k0**2 * jnp.where(
        mask, background_permitivitty - bead_permitivitty, jnp.zeros((3, 3))
    )

    return reduce(permitivitty, "(z nz) (y ny) (x nx) ni no -> z y x ni no", nz=antialiasing, ny=antialiasing, nx=antialiasing, reduction="mean")



def paper_sample() -> Array:
    # Simulation settings
    size = (4.55, 11.7, 11.7) # from paper
    spacing = 0.065 # [mum], from paper
    wavelength = 0.405 # [mum], from paper
    n_background = 1.33
    n_bead = jnp.array([1.44, 1.40, 1.37])  # z y x
    k0 = 2 * jnp.pi / wavelength
    bead_radius = 1.5 # [mum]

    # Calculating shape
    shape = np.around((np.array(size) / spacing)).astype(int) # without around becomes 1 less!

    # center of pixel is our coordinate
    z = jnp.linspace(1/2*spacing, size[0] - 1/2 * spacing, shape[0])
    y = jnp.linspace(size[1] - 1/2 * spacing, 1/2 * spacing, shape[1])
    x = jnp.linspace(1/2*spacing, size[2] - 1/2 * spacing, shape[2])
    grid = jnp.stack(jnp.meshgrid(z, y, x, indexing="ij"), axis=-1) 

    # Position of each bead, with radius 
    bead_pos = jnp.array([[size[0] / 2, 8.85, 2.85],
                      [size[0] / 2, 8.85, 8.85],
                      [size[0] / 2, 2.85, 2.85],
                      [size[0] / 2, 2.85, 8.85]])
    rotation = jnp.array([[0.0, jnp.pi/2, 0.0], 
                      [0.0, 0.0,0.0],
                      [0.0, 0.0, jnp.pi/2], 
                      [jnp.pi/4, jnp.pi/4, jnp.pi/4]])


    potential = jnp.zeros((*shape, 1, 3, 3))

    # Adding each bead
    for pos, orientation in zip(bead_pos, rotation):
        # Making bead and background
        bead_permitivitty = R(*orientation, inv=True).T @ jnp.diag(n_bead**2) @ R(*orientation, inv=True)
        background_permitivitty = jnp.eye(3) * n_background**2

        # Mask
        mask = jnp.sum((grid - pos)**2, axis=-1) < bead_radius ** 2

        # Making sample
        potential += (k0**2 * jnp.where(
            mask[..., None, None, None], background_permitivitty - bead_permitivitty, jnp.zeros((3, 3))
        ))

    return potential