#!/usr/bin/python3
"""
An example of inverse design that simply optimizes the scattering of a glass object towards two targets behind it, depending on the wavelength.

The script optimizes the refractive index to deposit as much light as possible into a target area. The starting point
is the same as example_solve2D, though refractive index can be varied between 1 and that in the example.
"""
import jax
import jax.numpy as jnp
import jaxopt
import jaxopt.linear_solve
import matplotlib.pyplot as plt

from convergent_born_iteration import electro_solver
from examples.convergent_born_series_solve2d import define_problem, display

def main():
    print(f'Starting {__name__} ...')
    grid, k0, permittivity, current_density, target_areas = define_problem([128, 128])

    def get_updated_permittivity(x):
        return 1j * permittivity.imag + 1 + (permittivity.real - 1) * jax.nn.sigmoid(x)

    def get_all_fields(permittivity):
        """
        TODO: implement implicit differentiation of the preconditioner.
        """
        return (electro_solver.solve(grid, k0 * (1 + 0.01 * _), permittivity, current_density, implicit_diff=False) for _ in range(2))


    def measure_intensity(x):
        updated_permittivity = get_updated_permittivity(x)

        def measure1(electric_fld, target_area):
            intensity = jnp.linalg.norm(electric_fld / 1e10, axis=0) ** 2  # TODO: pick a reasonable scale for light waves
            return jnp.vdot(target_area, intensity)

        return sum(measure1(fld, t) for fld, t in zip(get_all_fields(updated_permittivity), target_areas))

    # @jax.jit
    def loss(x):
        """The function to minimize iteratively."""
        return -measure_intensity(x)

    random_key = jax.random.key(0)
    x0 = jax.random.normal(random_key, permittivity.shape)
    x0 = x0 / jnp.linalg.norm(x0)

    initial_loss = loss(x0)
    print(f'Minimizing from loss {initial_loss:0.3f}...')
    verbose = True
    x, opt_state = jaxopt.GradientDescent(loss, value_and_grad=False, jit=True, maxiter=50, tol=1e-4, verbose=verbose).run(x0)  # Memory efficient but slow
    # x, opt_state = jaxopt.LBFGS(loss, value_and_grad=False, jit=True, verbose=verbose).run(x0)
    # x, opt_state = jaxopt.NonlinearCG(loss, value_and_grad=False, jit=True, max_stepsize=1, verbose=verbose).run(x0)
    # print(f'Executed {opt_state.num_fun_eval} function and {opt_state.num_grad_eval} evaluations.')

    # solver = optax.adam(learning_rate=1, eps=0.01)
    # x = x0.copy()
    # opt_state = solver.init(x)
    # for _ in range(15):
    #     updates, opt_state = solver.update(jax.grad(loss)(x), opt_state, x)
    #     x = optax.apply_updates(x, updates)

    print(f'Minimized loss from {initial_loss:0.3f} to {loss(x):0.3f}.')

    print('Solving optimized system and displaying...')
    permittivity = get_updated_permittivity(x)
    display(grid, permittivity, current_density, jnp.stack(list(get_all_fields(permittivity))), target_areas=target_areas)

    print('Done. Close figure window to exit.')
    plt.show()
    print('Exiting!')

if __name__ == '__main__':
    main()
