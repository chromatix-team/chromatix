#!/usr/bin/python3
"""
A script to measure the solver's time-efficiency using the example found in example_solve2d.py.
"""
import timeit

import jax.numpy as jnp

from convergent_born_iteration import log, electro_solver, example_solve2d

log.getChild(__name__)

def main():
    log.debug(f'Starting {__name__} ...')

    log.info('Defining the problem.')
    grid, k0, permittivity, current_density, _ = example_solve2d.define_problem([480, 640])

    log.info(f'Converting problem of shape {grid.shape} to JAX.')
    permittivity = jnp.array(permittivity)
    current_density = jnp.array(current_density)

    log.info('Solving repeatedly...')
    times = timeit.repeat(
        lambda: electro_solver.solve(grid, k0, permittivity, current_density, maxiter=1000, tol=0).block_until_ready(),
        repeat=10,
        number=1,
    )
    log.info(f'{min(times):0.3f}s is the best time out of {len(times)}. All times: {times}.')

    log.debug('Exiting!')

if __name__ == '__main__':
    main()
