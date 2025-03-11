#!/usr/bin/python3
"""
An example that computes the scattering of polarized waves off of an isotropic material in 2D. Extending this to 3D
should be trivial, bar the display code perhaps.

Some utility functions are imported from macromax. These are NumPy based, though these can be kept outside the core iteration.

There is already some code to handle anisotropy, though this remains untested and its initialization may be inefficient.

Magnetic and cross terms as used for chiral materials are not implemented as these add significantly more complexity.
"""
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from macromax.bound import LinearBound
from macromax.utils import Grid
from macromax.utils.display import complex2rgb, grid2extent

from convergent_born_iteration import log

log.getChild(__name__)

def define_problem(grid_shape = (128, ) * 2):
    """Define the electro-magnetic problem."""
    log.debug('Defining the top-level constants...')
    wavelength = 500e-9  # in meters
    plate_refractive_index = 1.5  # try making this negative (prepare to be patient though and avoid over sampling!)

    k0 = 2 * np.pi / wavelength
    grid = Grid(grid_shape, step=wavelength / 10)  # Dense sampling for a nicer display
    beam_diameter = grid.extent[1] / 4
    plate_thickness = grid.extent[0] / 4

    log.debug('Defining the boundary...')
    bound = LinearBound(grid, thickness=2e-6, max_extinction_coefficient=0.25)

    log.debug('Defining the incident wave...')
    incident_angle = 30 * np.pi / 180
    def rot_Z(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    incident_k = rot_Z(incident_angle) * k0 @ np.array([1, 0, 0])
    source_polarization = (rot_Z(incident_angle) @ np.array([0, 1, 1j]) / np.sqrt(2))[:, np.newaxis, np.newaxis]  # Add 2 dims on the right for a 2D grid
    current_density = (k0 ** 2) * np.exp(1j * (incident_k[0]*grid[0] + incident_k[1]*grid[1]))
    source_pixel = int(bound.thickness[0, 0] / grid.step[0])
    current_density[:source_pixel, :] = 0
    current_density[source_pixel+1:, :] = 0
    current_density = current_density * np.exp(-0.5*((grid[1] - grid[1].ravel()[grid.shape[0]//3])/(beam_diameter/2))**2)  # beam aperture
    current_density = source_polarization * current_density  # Make it vectorial by tagging on the polarization dimension on the left.

    log.debug('Defining the sample...')
    refractive_index = 1 + (plate_refractive_index - 1) * np.ones(grid[1].shape) * (np.abs(grid[0]) < plate_thickness/2)
    permittivity = refractive_index ** 2 + bound.electric_susceptibility  # just for clarity, macromax actually does this implicitly

    return grid, k0, permittivity, current_density

def display(grid, permittivity, current_density, E):
    """Display the input and output."""
    log.debug('Preparing the display...')
    fig, axs = plt.subplots(2, 3, sharex='all', sharey='all')
    structure = np.sqrt(permittivity) - 1
    structure /= np.amax(np.abs(structure))

    for axs_row, fields, label_pre in zip(axs,
                                          (structure + current_density / np.amax(np.abs(current_density)), E / np.amax(np.abs(E))),
                                          ('structure + $J_', '$E_'),
                                          ):
        for ax, field_component, ax_label in zip(axs_row, fields, 'xyz'):
            ax.imshow(complex2rgb(field_component), extent=grid2extent(grid / 1e-6))
            ax.set(xlabel=r'x  [$\mu$m]', ylabel=r'y  [$\mu$m]', title=label_pre + ax_label + '$')

def solve(grid_k, k0, permittivity, current_density):
    """
    Solve for the electromagnetic field.

    :param grid_k: A tuple with the (ifftshifted) k_space grid, corresponding to the wavevectors after FFT.
    :param k0: The vacuum wavenumber.
    :param permittivity: The (relative) permittivity distribution can either have the spatial dimensions, or it can have
        a 3x3 matrix in the first (left-most) axes, for each point in space.
    :param current_density: The current density, with the first (left-most) axis the polarization vector, while the
        remaining axes are spatial dimensions.

    :return: The electromagnetic field, E, with the first (left-most) axis the polarization vector, while the remaining
        axes are spatial dimensions.
    """
    isotropic = permittivity.ndim < len(grid_k) + 2 or permittivity.shape[0] == 1
    log.info(f"Solving for an {'' if isotropic else 'an'}isotropic material...")

    log.debug('Defining some functions to work only on the polarization or only on the spatial dimensions...')
    if isotropic:
        def norm(_):
            return jnp.abs(_)
    else:
        def norm(_):
            return jnp.linalg.norm(_, axis=(0, 1))  # This can probably be faster by a custom implementation (see macromax)

    def ft(x):
        """Fourier transform of the spatial dimensions."""
        return jnp.fft.fftn(x, axes=tuple(range(-len(grid_k), 0)))

    def ift(x):
        """Inverse Fourier transform of the spatial dimensions."""
        return jnp.fft.ifftn(x, axes=tuple(range(-len(grid_k), 0)))

    # The actual work is done here:
    log.debug('Scaling and shifting problem...')
    permittivity_bias = 1.2  # This can be chosen more optimally to minimize the following scaling factor, and maximize the convergence rate.
    scale = 1.1j * jnp.amax(norm(permittivity - permittivity_bias))  # Must be strictly larger than the norm in the polarization dimension
    assert jnp.amax((permittivity - permittivity_bias) / scale) < 1, f'Incorrect scale.'

    y = current_density / (k0 ** 2) / scale  # Use units so that k0 == 1 to avoid under/overflow
    log.debug(f'Using permittivity bias {permittivity_bias}, and scaling the whole problem by {scale}.')

    k2 = sum((_ / k0) ** 2 for _ in grid_k)  # Use units so that k0 == 1

    def split_trans_long_ft(x_ft):
        """Split a k-space vector field into its transverse and longitudinal components."""
        dc = k2 == 0  # to avoid division by 0
        projection_coefficient = sum(k * c for k, c in zip(grid_k, x_ft)) / (k2 + dc)
        x_long_ft = sum(k * projection_coefficient for k in grid_k) * dc
        x_trans_ft = x_ft - x_long_ft
        assert np.allclose(x_trans_ft + x_long_ft, x_ft), 'Transverse-longitudinal splitting failed!'
        return x_trans_ft, x_long_ft

    def forward(x):
        """The forward problem (for reference and testing)."""
        x_trans_ft, x_long_ft = split_trans_long_ft(ft(x))
        return (ift(x_trans_ft * (-k2 + permittivity_bias) + x_long_ft * permittivity_bias) +
                (permittivity - permittivity_bias) * x
                ) / scale

    def shifted_approx_inv(y):
        """The inverse of the scaled and shifted-by-1 approximation to the forward problem."""
        y_trans_ft, y_long_ft = split_trans_long_ft(ft(y))
        return ift(y_trans_ft / ((-k2 + permittivity_bias) / scale + 1) +
                     y_long_ft / (permittivity_bias / scale + 1)
                     )

    if isotropic:
        def shifted_discrepancy(x):
            """The scaled discrepancy, shifted by -1."""
            return ((permittivity - permittivity_bias) / scale - 1) * x
    else:
        def shifted_discrepancy(x):
            """The scaled discrepancy, shifted by -1."""
            return jnp.einsum('ij...,j...,i...', (permittivity - permittivity_bias) / scale - 1, x)

    log.info('Executing the preconditioned fixed point interation...')
    norm_y = jnp.linalg.norm(y)
    E = jnp.zeros_like(y)
    for iteration in range(1000):
        # The following tests are relatively slow, but these should not be executed in deployment
        assert jnp.amax(norm(shifted_discrepancy(E) + E)) < 1, f'The scaled discrepancy >= 1 !'
        assert jnp.vdot(E, forward(E)).real >= 0, 'Problem should not be dissipative!'

        dE = shifted_discrepancy(shifted_approx_inv(shifted_discrepancy(E) - y) + E)
        E = E + dE
        residue = jnp.linalg.norm(dE) / norm_y
        if iteration % 10 == 0:  # Report progress
            log.info(f'{iteration}: {residue:0.6f}')
        if residue < 1e-3:  # Stop criterion
            return E

def main():
    log.info(f'Starting {__name__} ...')

    grid, k0, permittivity, current_density = define_problem([256, 256])

    log.info('Transfer to GPU and solve...')
    dtype_float = jnp.float32
    dtype_complex = jnp.complex64
    grid_k = tuple(jnp.array(_, dtype_float) for _ in grid.k)
    permittivity = jnp.array(permittivity, dtype=dtype_complex)
    current_density = jnp.array(current_density, dtype=dtype_complex)
    E = solve(grid_k, k0, permittivity, current_density)

    display(grid, permittivity, current_density, E)

    log.info('Solved, close figure window to exit.')
    plt.show()
    log.debug('Exiting!')

if __name__ == '__main__':
    main()
