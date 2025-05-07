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


def multi_bead_sample(n_background: float, 
                      beads: list,  # list of dictionaries for each bead
                      shape: tuple, 
                      spacing: float, 
                      k0: float, 
                      antialiasing: int = 1):
    """
    Generate a sample containing multiple beads in a background medium.
    
    Parameters
    ----------
    n_background : float
        Refractive index of the background medium.
    beads : list of dict
        Each dict should contain:
          - "n_bead": ArrayLike of length 3 for the bead's refractive indices.
          - "orientation": ArrayLike of length 3 for the bead's rotation angles (radians).
          - "radius": float specifying the bead's radius.
          - "position": tuple or list of length 3 specifying the bead's center (in grid units).
    shape : tuple
        3-tuple specifying the shape of the sample volume (z, y, x).
    spacing : float
        Grid spacing in the sample volume.
    k0 : float
        Wavenumber in vacuum.
    antialiasing : int, optional
        Factor for antialiasing (default is 1).
        
    Returns
    -------
    Array
        4D array of shape (*shape, 3, 3) representing the sample. Each point
        in the 3D volume contains a 3x3 matrix representing the difference
        between the permittivity tensor of the bead(s) and the background.
    """
    # Adjust spacing for antialiasing.
    spacing = spacing / antialiasing

    # Create the high-resolution grid.
    grid = jnp.mgrid[: antialiasing * shape[0],
                      : antialiasing * shape[1],
                      : antialiasing * shape[2]]
    # Center the grid (this is our reference for positions).
    grid = grid - jnp.mean(grid, axis=(1, 2, 3), keepdims=True)

    # Initialize the sample with zeros.
    sample = jnp.zeros((*shape, 3, 3))
    # Define the background permittivity tensor.
    background_permittivity = jnp.eye(3) * n_background**2

    # Loop over each bead to add its contribution.
    for bead in beads:
        # Extract bead parameters.
        n_bead = bead["n_bead"]
        orientation = bead["orientation"]
        radius = bead["radius"]
        position = jnp.array(bead["position"])  # Expected to be in same units as grid

        # Shift the grid by the bead's center.
        # Note: We need to reshape position to broadcast properly over grid dimensions.
        pos_reshaped = position[:, None, None, None]
        grid_shifted = grid - pos_reshaped

        # Create the spherical mask for this bead.
        mask = jnp.sum(grid_shifted**2, axis=0) < (radius / spacing)**2

        # Compute the bead's permittivity tensor.
        bead_permittivity = R(*orientation).T @ jnp.diag(jnp.array(n_bead)**2) @ R(*orientation)
        
        # Compute the bead's contribution.
        contribution = k0**2 * jnp.where(mask[..., None, None], background_permittivity - bead_permittivity, 0)

        # Add the contribution to the sample.
        sample = sample + reduce(contribution,
                                 "(z nz) (y ny) (x nx) ni no -> z y x ni no",
                                 nz=antialiasing, ny=antialiasing, nx=antialiasing,
                                 reduction="mean")
    return sample



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
        bead_permitivitty = R(*orientation).T @ jnp.diag(n_bead**2) @ R(*orientation)
        background_permitivitty = jnp.eye(3) * n_background**2

        # Mask
        mask = jnp.sum((grid - pos)**2, axis=-1) < bead_radius ** 2

        # Making sample
        potential += (k0**2 * jnp.where(
            mask[..., None, None, None], background_permitivitty - bead_permitivitty, jnp.zeros((3, 3))
        ))

    return potential


def principal_dielectric_tensor_single(n_o: float, n_e: float) -> jnp.ndarray:
    """
    Returns the principal dielectric tensor for a uniaxial material.
    :param n_o: Ordinary refractive index
    :param n_e: Extraordinary refractive index
    :return: 3x3 principal dielectric tensor
    """
    return jnp.diag(jnp.array([n_e**2, n_o**2, n_o**2]))


def principal_dielectric_tensor(n_o: ArrayLike, n_e: ArrayLike) -> jnp.ndarray:
    """
    Returns the principal dielectric tensors for a uniaxial material across a 3D volume.
    :param n_o: 3D array of ordinary refractive indices with shape (z, y, x)
    :param n_e: 3D array of extraordinary refractive indices with shape (z, y, x)
    :return: 5D array of shape (z, y, x, 3, 3) containing 3x3 principal dielectric tensors for each voxel.
    """
    # Ensure n_o and n_e are arrays
    n_o = jnp.asarray(n_o)
    n_e = jnp.asarray(n_e)
    
    # Check if the input arrays have the same shape
    if n_o.shape != n_e.shape:
        raise ValueError("n_o and n_e must have the same shape (z, y, x)")

    # Create a 3D tensor field with shape (z, y, x, 3, 3)
    tensors = jnp.zeros(n_o.shape + (3, 3), dtype=n_o.dtype)
    
    # Fill the diagonal components
    tensors = tensors.at[..., 0, 0].set(n_e**2)  # Extraordinary index on the first component
    tensors = tensors.at[..., 1, 1].set(n_o**2)  # Ordinary index on the second component
    tensors = tensors.at[..., 2, 2].set(n_o**2)  # Ordinary index on the third component

    return tensors


def dielectric_tensor_global(n_o: float, n_e: float, optic_axis: ArrayLike) -> jnp.ndarray:
    """
    Constructs the global dielectric tensor for a uniaxial material by rotating the principal dielectric tensor.
    :param n_o: Ordinary refractive index
    :param n_e: Extraordinary refractive index
    :param optic_axis: Optic axis vector [a_z, a_y, a_x]
    :return: 3x3 dielectric tensor in global coordinates
    """
    # Principal dielectric tensor in the local frame
    epsilon_prime = principal_dielectric_tensor(n_o, n_e)
    
    # Rotation matrix from local to global coordinates
    rotation_matrices = rotation_matrix_to_align_axes(optic_axis)
    
    # # Apply rotation: ε = R^T * ε' * R
    # epsilon_global = R_global.T @ epsilon_prime @ R_global

    # Vectorized matrix multiplication: ε_global = R^T * ε' * R
    epsilon_global = jnp.einsum(
        '...ij,...jk,...lk->...il',
        rotation_matrices.transpose(0, 1, 2, 4, 3),  # R^T
        epsilon_prime,
        rotation_matrices  # R
    )
    
    return epsilon_global


def rotation_matrix_to_align_axes_single(optic_axis: ArrayLike) -> np.ndarray:
    """
    Construct a rotation matrix that aligns the z-axis with the given unit vector optic_axis.
    :param optic_axis: Unit vector [a_z, a_y, a_x] as the target z-axis.
    :return: 3x3 rotation matrix.
    """
    # Ensure optic_axis is a unit vector
    optic_axis = optic_axis / np.linalg.norm(optic_axis)

    # Create a vector orthogonal to optic_axis for the new x-axis
    if not np.isclose(optic_axis[0], 1):  # Avoid `a_z` close to ±1 to prevent numerical issues
        x_new = np.array([-optic_axis[1], optic_axis[0], 0])
        x_new = x_new / np.linalg.norm(x_new)
    else:
        # If optic_axis is very close to [1, 0, 0] or [-1, 0, 0], choose a standard orthogonal vector
        x_new = np.array([0, 0, 1])
    
    # Create the new y-axis using the cross product
    y_new = np.cross(x_new, optic_axis)
    
    # Normalize the new y-axis
    y_new = y_new / np.linalg.norm(y_new)
    
    # Construct the rotation matrix with the axes in [z, y, x] order as columns
    R = np.array([optic_axis, y_new, x_new]).T  # Transpose to align columns with the new basis

    return R


def rotation_matrix_to_align_axes(optic_axis: ArrayLike) -> np.ndarray:
    """
    Construct a rotation matrix that aligns the z-axis with the given unit vector optic_axis.
    :param optic_axis: Unit vector [a_z, a_y, a_x] as the target z-axis.
    :return: Array of shape (z, y, x, 3, 3) representing the rotation matrices for each voxel.
    """
    # Calculate the norm of the optic_axis to identify zero vectors
    norm = jnp.linalg.norm(optic_axis, axis=0, keepdims=False)
    is_zero_vector = jnp.isclose(norm, 0)
    optic_axis = jnp.where(is_zero_vector, 1.0, optic_axis / norm)  # Normalize where non-zero, leave as 1 where zero

    # Create a default orthogonal vector for each voxel
    x_new = jnp.stack([-optic_axis[1], optic_axis[0], jnp.zeros_like(optic_axis[0])], axis=0)
    x_new_norm = jnp.linalg.norm(x_new, axis=0, keepdims=True)
    x_new = jnp.where(is_zero_vector, 1.0, x_new / x_new_norm)  # Avoid normalization issues for zero vectors

    # Create the y_new vector using the cross product
    y_new = jnp.cross(x_new, optic_axis, axisa=0, axisb=0, axisc=0)
    y_new_norm = jnp.linalg.norm(y_new, axis=0, keepdims=True)
    y_new = jnp.where(is_zero_vector, 1.0, y_new / y_new_norm)  # Normalize where non-zero

    # Construct the rotation matrices and set identity matrices for zero vectors
    R = jnp.stack([optic_axis, y_new, x_new], axis=-1).transpose(1, 2, 3, 0, 4)
    identity_matrices = jnp.eye(3) #.reshape(1, 1, 1, 3, 3)  # Shape (1, 1, 1, 3, 3)
    R = jnp.where(is_zero_vector[..., None, None], identity_matrices, R)  # Replace with identity matrix where zero
    
    return R


def calc_extraordinary_refractive_index(birefringence: ArrayLike, n_o: float) -> ArrayLike:
    return n_o + birefringence
