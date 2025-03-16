#!/usr/bin/python3
"""
An example of inverse design that simply optimizes the scattering of a glass mask towards a target behind it.

The script optimizes the refractive index to deposit as much light as possible into a target area. The starting point
is the same as example_solve2D, though refractive index can be varied between 1 and that in the example.
"""
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jaxopt
import jaxopt.linear_solve

from convergent_born_iteration import log, electro_solver, example_solve2d

log.getChild(__name__)

def main():
    log.info(f'Starting {__name__} ...')

    grid, k0, permittivity, current_density, target_area = example_solve2d.define_problem([128, 128])

    log.info(f'Converting problem of shape {grid.shape} to JAX.')
    grid_k = tuple(jnp.array(_) for _ in grid.k)
    permittivity = jnp.array(permittivity)
    current_density = jnp.array(current_density)
    target_area = jnp.array(target_area)

    def get_updated_permittivity(x):
        return 1j * permittivity.imag + 1 + (permittivity.real - 1) * jax.nn.sigmoid(x)

    def measure_intensity(x):
        E = electro_solver.solve(grid_k, k0, get_updated_permittivity(x), current_density, implicit_diff=False)  # TODO: implement implicit differentiation of the preconditioner.
        I = jnp.linalg.norm(E, axis=0) ** 2
        return jnp.vdot(target_area, I)

    @jax.jit
    def loss(x):
        """The function to minimize iteratively."""
        return -measure_intensity(x)

    random_key = jax.random.key(0)
    x0 = jax.random.normal(random_key, current_density.shape)  # use shape[-1:] to optimize in 1D, and have enough memory for LBFGS and NonLinearCG
    x0 = x0 / jnp.linalg.norm(x0)

    initial_loss = loss(x0)
    log.info(f'Minimizing from loss {initial_loss:0.3f}...')
    verbose = True
    x, opt_state = jaxopt.GradientDescent(loss, value_and_grad=False, jit=True, tol=0.01, verbose=verbose).run(x0)  # Memory efficient but slow
    # x, opt_state = jaxopt.LBFGS(loss, value_and_grad=False, jit=True, verbose=verbose).run(x0)
    # x, opt_state = jaxopt.NonlinearCG(loss, value_and_grad=False, jit=True, max_stepsize=1, verbose=verbose).run(x0)
    # log.info(f'Executed {opt_state.num_fun_eval} function and {opt_state.num_grad_eval} evaluations.')
    log.info(f'Minimized loss from {initial_loss:0.3f} to {loss(x):0.3f}.')

    log.info('Solving optimized system...')
    E = electro_solver.solve(grid_k, k0, get_updated_permittivity(x), current_density, implicit_diff=False)

    log.info('Displaying...')
    example_solve2d.display(grid, get_updated_permittivity(x), current_density, target_area, E=E)

    log.info('Done. Close figure window to exit.')
    plt.show()
    log.debug('Exiting!')

if __name__ == '__main__':
    main()
