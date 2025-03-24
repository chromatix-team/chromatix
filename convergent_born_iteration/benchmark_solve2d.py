#!/usr/bin/python3
"""
A script to measure the solver's time-efficiency using the example found in example_solve2d.py.
"""
import timeit

import jax
import jax.numpy as jnp

from convergent_born_iteration import electro_solver
from examples.convergent_born_series_solve2d import define_problem


def main():
    print(f'Starting {__name__} ...')

    print('Defining the problem.')
    grid, k0, permittivity, current_density, _ = define_problem([480, 640])

    print(f'Converting problem of shape {grid.shape} to JAX.')
    permittivity = jnp.array(permittivity)
    current_density = jnp.array(current_density)

    print('JIT-ing the solver...')
    @jax.jit
    def solve(permittivity, current_density, maxiter, tol):
        return electro_solver.solve(grid, k0, permittivity, current_density, maxiter=maxiter, tol=tol)

    print('Solving repeatedly...')
    times = timeit.repeat(
        lambda: solve(permittivity, current_density, maxiter=1000, tol=0).block_until_ready(),
        repeat=10,
        number=1,
    )
    print(f'{min(times):0.3f}s is the best time out of {len(times)}. All times: {times}.')

    print('Exiting!')

if __name__ == '__main__':
    main()
