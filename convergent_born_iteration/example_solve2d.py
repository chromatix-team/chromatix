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
import jax.numpy as jnp

from macromax.bound import LinearBound
from macromax.utils import Grid
from macromax.utils.display import complex2rgb, grid2extent

from convergent_born_iteration import log, electro_solver

log.getChild(__name__)

def define_problem(grid_shape = (256, 256)):
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
    current_density = np.exp(1j * (incident_k[0]*grid[0] + incident_k[1]*grid[1]))
    source_pixel = int(bound.thickness[0, 0] / grid.step[0])
    current_density[:source_pixel, :] = 0
    current_density[source_pixel+1:, :] = 0
    current_density = current_density * np.exp(-0.5*((grid[1] - grid[1].ravel()[grid.shape[1]//3])/(beam_diameter/2))**2)  # beam aperture
    current_density = source_polarization * current_density  # Make it vectorial by tagging on the polarization dimension on the left.

    log.debug('Defining the sample...')
    refractive_index = 1 + (plate_refractive_index - 1) * np.ones(grid[1].shape) * (np.abs(grid[0]) < plate_thickness/2)
    permittivity = refractive_index ** 2 + bound.electric_susceptibility  # just for clarity, macromax actually does this implicitly

    target_radius = max(min(grid.extent) / 64, min(grid.step))
    target_area = sum((rng - o) ** 2 for rng, o in zip(grid, (grid.first[0] + grid.extent[0] - bound.thickness[0, 1], 0))) < target_radius ** 2

    return grid, k0, permittivity, current_density, target_area

def display(grid, permittivity, current_density, target_area=0.0, E=0.0):
    """Display the input and output, including absorbing boundaries for clarity."""
    log.debug('Preparing the display...')
    fig, axs = plt.subplots(2, 3, sharex='all', sharey='all')
    structure = np.sqrt(permittivity) - 1 - 0.5j * target_area
    structure /= np.amax(np.abs(structure))

    for axs_row, fields, label_pre in zip(axs,
                                          (structure + current_density / np.amax(np.abs(current_density)), E / np.amax(np.abs(E))),
                                          ('structure + $J_', '$E_'),
                                          ):
        for ax, field_component, ax_label in zip(axs_row, fields, 'xyz'):
            ax.imshow(complex2rgb(field_component), extent=grid2extent(grid / 1e-6))
            ax.set(xlabel=r'x  [$\mu$m]', ylabel=r'y  [$\mu$m]', title=label_pre + ax_label + '$')

def main():
    log.debug(f'Starting {__name__} ...')

    log.info('Defining the problem.')
    grid, k0, permittivity, current_density, _ = define_problem([480, 640])

    log.info('Converting to JAX.')
    grid_k = tuple(jnp.array(_) for _ in grid.k)
    permittivity = jnp.array(permittivity)
    current_density = jnp.array(current_density)

    log.info('Solving...')
    E = electro_solver.solve(grid_k, k0, permittivity, current_density, implicit_diff=True)

    log.info('Displaying...')
    display(grid, permittivity, current_density, E=E)

    log.info('Done. Close figure window to exit.')
    plt.show()
    log.debug('Exiting!')

if __name__ == '__main__':
    main()
