# %% Imports
from typing import Optional, Tuple

import imageio
import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import imageio
import numpy as np


def generate_permittivity_tensor(
    n_o: float, n_e: float, extraordinary_axis: Optional[str] = "x"
):
    """
    Generate the permittivity tensor for a uniaxial anisotropic material.

    Args:
        n_o (float): Ordinary refractive index
        n_e (float): Extraordinary refractive index
        extraordinary_axis (str): Axis which is extraordinary ('x', 'y', or 'z')

    Returns:
        jnp.ndarray: Permittivity tensor with the order of axes as zyx
    """
    epsilon_o = n_o**2
    epsilon_e = n_e**2
    if extraordinary_axis == "z":
        epsilon_tensor = jnp.array(
            [[epsilon_e, 0, 0], [0, epsilon_o, 0], [0, 0, epsilon_o]]
        )
    elif extraordinary_axis == "y":
        epsilon_tensor = jnp.array(
            [[epsilon_o, 0, 0], [0, epsilon_e, 0], [0, 0, epsilon_o]]
        )
    elif extraordinary_axis == "x":
        epsilon_tensor = jnp.array(
            [[epsilon_o, 0, 0], [0, epsilon_o, 0], [0, 0, epsilon_e]]
        )
    else:
        raise ValueError("extraordinary_axis must be one of 'x', 'y', or 'z'")
    return epsilon_tensor


def create_homogeneous_phantom(
    shape: Tuple[int, int, int],
    n_o: float,
    n_e: float,
    extraordinary_axis: Optional[str] = "x",
):
    """
    Create a homogeneous uniaxial anisotropic phantom.

    Args:
        shape (tuple): Shape of the phantom (z, y, x)
        n_o (float): Ordinary refractive index
        n_e (float): Extraordinary refractive index
        extraordinary_axis (str): Axis which is extraordinary ('x', 'y', or 'z')

    Returns:
        jnp.ndarray: 4D array representing the phantom with the
                    permittivity tensor at each voxel
    """
    epsilon_tensor = generate_permittivity_tensor(n_o, n_e, extraordinary_axis)
    phantom = jnp.tile(epsilon_tensor, (*shape, 1, 1))
    return phantom


def create_calcite_crystal(
    shape: Tuple[int, int, int], extraordinary_axis: Optional[str] = "z"
):
    """
    Create a calcite crystal phantom.

    Args:
        shape (tuple): Shape of the phantom (z, y, x)
        extraordinary_axis (str): Axis which is extraordinary ('x', 'y', or 'z')

    Returns:
        jnp.ndarray: 4D array representing the phantom with the
                    permittivity tensor at each voxel
    """
    n_o = 1.658
    n_e = 1.486
    return create_homogeneous_phantom(shape, n_o, n_e, extraordinary_axis)


# %% Scattering potential functions
def create_scattering_potential(permittivity_tensor, background_permittivity):
    """
    Create the scattering potential from the permittivity tensor.

    Args:
        permittivity_tensor (jnp.ndarray): The permittivity tensor of the material.
        background_permittivity (float): The permittivity of the background medium.

    Returns:
        jnp.ndarray: The scattering potential.
    """
    # Calculate the permittivity contrast
    contrast = permittivity_tensor - background_permittivity

    # Scattering potential is proportional to the permittivity contrast
    scattering_potential = contrast / background_permittivity

    return scattering_potential


def permittivity_tensor_from_pixel(
    pixel_value, n_o_base=1.55, n_e_base=1.55, scale=0.5
):
    # The difference between n_o and n_e increases with the pixel value
    n_o = n_o_base + scale * pixel_value
    n_e = n_e_base - scale * pixel_value
    return generate_permittivity_tensor(n_o, n_e)


def vectorized_permittivity_tensor_from_pixel(
    img, n_o_base=1.55, n_e_base=1.55, scale=0.5
):
    vmap_func = jax.vmap(
        lambda pixel: permittivity_tensor_from_pixel(pixel, n_o_base, n_e_base, scale)
    )
    return jax.vmap(vmap_func)(img)


def create_homogeneous_scattering_potential(
    shape: Tuple[int, int, int], n_o: float, n_e: float, background_permittivity: float
):
    """
    Create a homogeneous uniaxial anisotropic scattering potential.

    Args:
        shape (tuple): Shape of the phantom (z, y, x)
        n_o (float): Ordinary refractive index
        n_e (float): Extraordinary refractive index
        background_permittivity (float): Background permittivity

    Returns:
        jnp.ndarray: 4D array representing the scattering potential
    """
    permittivity_tensor = create_homogeneous_phantom(shape, n_o, n_e)
    scattering_potential = create_scattering_potential(
        permittivity_tensor, background_permittivity
    )
    return scattering_potential


def calc_scattering_potential(epsilon_r, refractive_index, wavelength):
    """
    Create the scattering potential from the permittivity tensor.

    Args:
        epsilon_r (jnp.ndarray): The permittivity tensor of the material.
        refractive_index (float): The refractive index of the background medium.
        wavelength (float): The wavelength of the light (microns).

    Returns:
        jnp.ndarray: The scattering potential.
    """
    k_0 = 2 * jnp.pi / wavelength
    vol_shape = epsilon_r.shape[:3]
    epsilon_m = jnp.tile(jnp.eye(3) * refractive_index**2, (*vol_shape, 1, 1))
    scattering_potential = k_0**2 * (epsilon_m - epsilon_r)
    return scattering_potential


def calc_scattering_potential_unscaled(epsilon_r, refractive_index):
    """
    Create the scattering potential from the permittivity tensor without
    scaling by k_0^2 which depends on the wavelength.

    Args:
        epsilon_r (jnp.ndarray): The permittivity tensor of the material.
        refractive_index (float): The refractive index of the background medium.

    Returns:
        jnp.ndarray: The scattering potential without scaling factor.
    """
    vol_shape = epsilon_r.shape[:-2]
    epsilon_m = jnp.tile(jnp.eye(3) * refractive_index**2, (*vol_shape, 1, 1))
    scattering_potential_unscaled = epsilon_m - epsilon_r
    return scattering_potential_unscaled


def process_image_to_epsilon_r(input_path, n_o=1.658, n_e=1.486):
    img = imageio.imread(input_path)
    img = img / img.max()
    jax_img = jnp.array(img)

    n_avg = (n_o + n_e) / 2
    scale = (n_o - n_e) / 2
    epsilon_img = vectorized_permittivity_tensor_from_pixel(
        jax_img, n_avg, n_avg, scale
    )

    # Tile the epsilon tensor
    epsilon_r = jnp.tile(epsilon_img, (10, 1, 1, 1, 1))

    return epsilon_r


def expand_potential_dims(tensor):
    potential = jnp.expand_dims(tensor, axis=(1, 4))
    return potential


def generate_dummy_potential(vol_shape):
    potential = expand_potential_dims(jnp.ones((*vol_shape, 3, 3)))
    return potential


def form_diag_permittivity(n_o, n_e):
    """
    Create a diagonal permittivity tensor for a uniaxial crystal.

    Args:
        n_o (float): The ordinary refractive index.
        n_e (float): The extraordinary refractive index.

    Returns:
        np.ndarray: A 3x3 diagonal permittivity tensor.
    """
    return np.diag([n_o**2, n_o**2, n_e**2])


def create_rotation_matrix(v):
    """
    Generates a rotation matrix that aligns the z-axis with the given unit vector v.

    Args:
        v (array-like): A 3D unit vector [v_x, v_y, v_z].

    Returns:
        jnp.ndarray: A 3x3 rotation matrix.
    """
    v = jnp.asarray(v)

    if not jnp.isclose(jnp.linalg.norm(v), 1.0):
        raise ValueError("The input vector is not a unit vector.")

    # Ensure the vector is a unit vector
    v = v / jnp.linalg.norm(v)

    # If the vector is already aligned with the z-axis
    if jnp.allclose(v, jnp.array([0, 0, 1])):
        return jnp.eye(3)
    elif jnp.allclose(v, jnp.array([0, 0, -1])):
        return jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Compute the cross product and angle
    z_axis = jnp.array([0, 0, 1])
    v_cross_z = jnp.cross(z_axis, v)
    sin_theta = jnp.linalg.norm(v_cross_z)
    cos_theta = jnp.dot(z_axis, v)

    # Normalize the rotation axis
    k = v_cross_z / sin_theta

    # Construct the skew-symmetric matrix for the rotation axis
    K = jnp.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

    # Construct the rotation matrix using Rodrigues' rotation formula
    I = jnp.eye(3)
    rotation_matrix = I + sin_theta * K + (1 - cos_theta) * jnp.dot(K, K)

    return rotation_matrix


def permittivity_from_vector(n_o, n_e, v):
    """
    Create a permittivity tensor for a uniaxial crystal with the extraordinary axis aligned with the given vector.

    Args:
        n_o (float): The ordinary refractive index.
        n_e (float): The extraordinary refractive index.
        v (array-like): A 3D unit vector [v_x, v_y, v_z].

    Returns:
        jnp.array: A 3x3 permittivity tensor.
    """
    rot_matrix = create_rotation_matrix(v)
    diagonal_tensor = form_diag_permittivity(n_o, n_e)
    permittivity_tensor = rot_matrix @ diagonal_tensor @ rot_matrix.T
    return permittivity_tensor
