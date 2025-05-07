# %% imports
from __future__ import annotations

import jax.numpy as jnp
from samples import Sample
from scipy.special import expi
from solvers import helmholtz_solver


def theoretical_field(k, x, dx):
    k_plus = k + jnp.pi / dx
    k_min = k - jnp.pi / dx
    x = jnp.maximum(x, 1e-12)  # fixing x = 0

    # Calculating field
    first_term = 2 * 1j * jnp.pi + expi(1j * k_min * x) - expi(1j * k_plus * x)
    first_term *= dx / (4 * k * jnp.pi) * jnp.exp(-1j * k * x)

    second_term = expi(-1j * k_min * x) - expi(-1j * k_plus * x)
    second_term *= dx / (4 * k * jnp.pi) * jnp.exp(1j * k * x)

    return first_term + second_term


# %% Settings and sample
lambda_ = 1.0
dx = lambda_ / 4
boundary_strength = 0.2
boundary_thickness = (None, 25)

shape = (1, 200)
n_sample = 1.33

source = jnp.zeros(shape, dtype=jnp.complex64).at[:, 0].set(1.0)
n_sample = jnp.full(shape, n_sample)
sample = Sample.init(n_sample, source, dx, lambda_, boundary_thickness)


# %%

x = jnp.arange(200) * dx
k = 2 * jnp.pi * n_sample / lambda_
phi_a, limit = theoretical_field(k, x, dx)
field, stats = helmholtz_solver(sample)
print(stats["n_iterations"])
assert jnp.mean(jnp.abs(field.squeeze() - phi_a) ** 2) < 1e-9
# %%
