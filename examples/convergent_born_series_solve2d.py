#!/usr/bin/python3
"""
An example that computes the scattering of polarized waves off of an isotropic material in 2D. Extending this to 3D
should be trivial, bar the display code perhaps.

Some utility functions are imported from macromax. These are NumPy based, though these can be kept outside the core iteration.

There is already some code to handle anisotropy, though this remains untested and its initialization may be inefficient.

Magnetic and cross terms as used for chiral materials are not implemented as these add significantly more complexity.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from chromatix.utils import Grid, dim
from chromatix.utils.display import complex2rgb, grid2extent
from convergent_born_iteration import electro_solver
from convergent_born_iteration.bound import LinearBound

def define_problem(grid_shape = (256, 256)):
    """Define the electro-magnetic problem."""
    print('Defining the top-level constants...')
    wavelength = 500e-9  # in meters
    material_refractive_index = 1.5  # try making this negative (prepare to be patient though and avoid over sampling!)

    k0 = 2 * jnp.pi / wavelength
    grid = Grid(grid_shape, step=wavelength / 8)  # Overly-dense sampling for a nicer display
    beam_diameter = grid.extent[0] / 16
    object_diameter = grid.extent[0] / 2

    print('Defining the boundary...')
    bound = LinearBound(grid, thickness=2e-6, max_extinction_coefficient=0.25)

    print('Defining the incident wave...')
    incident_angle = 90 * jnp.pi / 180
    def rot_Z(a): return jnp.array([[jnp.cos(a), -jnp.sin(a), 0], [jnp.sin(a), jnp.cos(a), 0], [0, 0, 1]])
    incident_k = rot_Z(incident_angle) * k0 @ jnp.array([1, 0, 0])
    source_polarization = dim.add(rot_Z(incident_angle) @ jnp.array([0, 1, 1j]) / jnp.sqrt(2), right=grid.ndim)  # Add dims on the right
    current_density = jnp.exp(1j * sum(k * g for k, g in zip(incident_k, grid)))
    source_pixel_index = int(bound.thickness[1, 0] / grid.step[1])
    current_density = current_density * (jnp.arange(current_density.shape[-1]) == source_pixel_index)
    current_density = current_density * jnp.exp(-0.5* ((grid[0] - object_diameter / 2) / (beam_diameter/2)) ** 2)  # beam aperture
    current_density = source_polarization * current_density  # Make it vectorial by tagging on the polarization dimension on the left.

    print('Defining the sample...')
    # refractive_index = 1 + (material_refractive_index - 1) * jnp.ones(grid[1].shape) * (jnp.abs(grid[0]) < object_diameter/2)
    refractive_index = 1 + (material_refractive_index - 1) * (sum(_ ** 2 for _ in grid) ** 0.5 < object_diameter / 2)
    permittivity = refractive_index ** 2 + bound.electric_susceptibility  # just for clarity, macromax actually does this implicitly

    # A simple test for anisotropy
    # permittivity = dim.add(jnp.eye(3), right=grid.ndim) * permittivity

    target_radius = max(min(grid.extent) / 64, min(grid.step))
    target_areas = jnp.stack(
        [
            sum((rng - o) ** 2 for rng, o in zip(grid, (grid.first[0] + bound.thickness[0, 1], grid.first[1] + grid.extent[1] - bound.thickness[1, 1]))) < target_radius ** 2,
            sum((rng - o) ** 2 for rng, o in zip(grid, (grid.first[0] + bound.thickness[0, 1], grid.first[1] + bound.thickness[1, 0]))) < target_radius ** 2,
            ]
    )

    print(f'Defined a problem of shape {grid.shape}.')

    return grid, k0, permittivity, current_density, target_areas

def display(grid, permittivity, current_density, E, labels=None, target_areas=(0, 0)):
    """Display the input and output, including absorbing boundaries for clarity."""
    if E.ndim <= current_density.ndim:  # Check if multiple inputs
        E = E[jnp.newaxis]
    labels = ('' for _ in range(len(E))) if labels is None else (_ + ' | ' for _ in labels)
    fig, axs = plt.subplots(1 + E.shape[0], 3, sharex='all', sharey='all')
    structure = jnp.sqrt(permittivity) - 1 - (-0.5 + 0.5j) * jnp.diff(jnp.asarray(target_areas, dtype=jnp.float32), axis=0)
    structure /= jnp.amax(jnp.abs(structure))

    for axs_row, fld, label_pre in zip(axs,
                                       (structure + current_density / jnp.amax(jnp.abs(current_density)),
                                        *(_ / jnp.amax(jnp.abs(_)) for _ in E),
                                        ),
                                       ('structure + $J_', *(_ + '$E_' for _ in labels)),
                                       ):
        normalization = jnp.amax(jnp.abs(fld)) / 1.5
        for ax, fld_component, ax_label in zip(axs_row, fld, 'xyz'):
            if fld_component.ndim > 2:
                fld_component = fld_component[..., fld_component.shape[-1] // 2]
            if grid.ndim > 2:
                grid = grid.project(axes_to_remove=-1)
            ax.imshow(complex2rgb(fld_component / normalization), extent=grid2extent(grid / 1e-6))
            ax.set(xlabel=r'x  [$\mu$m]', ylabel=r'y  [$\mu$m]', title=label_pre + ax_label + '$ ')

def main():
    print(f'Starting {__name__} ...')
    grid, k0, permittivity, current_density, _ = define_problem([256, 256 * 4 // 3])

    print('Solving...')
    @jax.jit
    def solve(permittivity, current_density):
        return electro_solver.solve(grid, k0, permittivity, current_density)
    E = solve(permittivity, current_density)

    print('Displaying...')
    display(grid, permittivity, current_density, E=E)

    print('Done. Close figure window to exit.')
    plt.show()

if __name__ == '__main__':
    main()
