from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from samples import Sample


def helmholtz_solver(sample: Sample, rtol=1e-9, max_iter: int = 1000):
    def update_fn(args):
        field, history, iteration = args

        # New field
        dE = field - jnp.fft.ifftn(Gk * jnp.fft.fftn(V * field + source))
        field = field - 1j / epsilon * V * dE

        # Calculating change
        delta = jnp.mean(jnp.abs(dE[sample.roi]) ** 2)
        delta /= jnp.mean(jnp.abs(field[sample.roi]) ** 2)

        return field, history.at[iteration].set(delta), iteration + 1

    def cond_fn(args) -> bool:
        _, history, iteration = args
        # return (history[iteration - 1] > rtol) & (iteration < max_iter)
        return iteration < max_iter

    # Unpacking sample
    source, V, epsilon = sample.source, sample.potential(), sample.epsilon

    # Making Greens vector
    k_red_sq = sample.km_sq + 1j * epsilon
    Gk = 1 / (4 * jnp.pi**2 * jnp.sum(sample.k_grid**2, axis=0) - k_red_sq)

    # Running and postprocessing
    init = update_fn((source, jnp.zeros(max_iter), 0))
    field, history, iteration = jax.lax.while_loop(cond_fn, update_fn, init)
    return field[sample.roi], {"error": history, "n_iterations": iteration}


# %% Making Greens function
def G_fn(k: Array, k0: Array, alpha: Array) -> Array:
    #k_sq = jnp.sum(jnp.abs(k) ** 2, axis=-1)[..., None, None]
    #k_cross = k[..., :, None] * k[..., None, :] / (alpha * k0**2)
    #return (jnp.eye(3) - k_cross) / (k_sq - alpha * k0**2)
    k_sq =  jnp.sum(jnp.abs(k) ** 2, axis=-1)[..., None, None]
    pi_L = k[..., :, None] * k[..., None, :] / k_sq
    pi_L = jnp.where(k[..., :, None] ==0, 0, pi_L)
    pi_t = jnp.eye(3) - pi_L
    G = pi_t / (k_sq - alpha * k0**2) - pi_L / (alpha * k0**2)
    return G


def bmatvec(mat: Array, vec: Array) -> Array:
    return jnp.matmul(mat, vec[..., None]).squeeze(-1)


def propagate(G: Array, field: Array) -> Array:
    fft = lambda x: jnp.fft.fftn(x, axes=(0, 1, 2))
    ifft = lambda x: jnp.fft.ifftn(x, axes=(0, 1, 2))

    return ifft(bmatvec(G, fft(field)))


# %%
def maxwell_solver(source: Source, sample: Sample, rtol=1e-8, max_iter: int = 1000):
    def update_fn(args):
        field, history, iteration = args

        # New field
        field_prop = propagate(Gk, k0**2 * bmatvec(xi, field) + _source)
        dE = 1j / alpha_imag * bmatvec(xi, (field_prop - field))

        # Calculating change
        delta = jnp.mean(jnp.abs(dE) ** 2) / jnp.mean(jnp.abs(field) ** 2)

        return field + dE, history.at[iteration].set(delta), iteration + 1

    def cond_fn(args) -> bool:
        _, history, iteration = args
        return (history[iteration - 1] > rtol) & (iteration < max_iter)

    # Getting real part of alpha
    alpha_real = (jnp.min(sample.permittivity) + jnp.max(sample.permittivity)) / 2
    alpha_imag = jnp.max(jnp.abs(sample.permittivity - alpha_real)) / 0.95
    alpha = alpha_real + 1j * alpha_imag

    # Making greens function and potential
    k0 = 2 * jnp.pi / source.wavelength
    Gk = G_fn(sample.k_grid, k0, alpha)
    xi = sample.permittivity - alpha * jnp.eye(3)

    # Setting up source
    _source = (
        jnp.zeros((*sample.spatial_shape, 3), dtype=jnp.complex64)
        .at[*sample.roi, :]
        .set(source.source)
    )

    # Running and postprocessing
    init = update_fn((jnp.zeros_like(_source), jnp.zeros(max_iter), 0))
    field, history, iteration = jax.lax.while_loop(cond_fn, update_fn, init)
    return field[sample.roi], {
        "full_field": field,
        "error": history,
        "n_iterations": iteration,
    }
