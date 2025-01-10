# %% imports
from functools import reduce

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import expi, factorial

# %% Settings and sample

lambda_ = 1.0
dx = lambda_ / 4
k = 2 * jnp.pi / lambda_
beta = 0.2
boundary_thickness = 25
shape = (1, 200)
x = jnp.arange(200) * dx


# %% Making theoretical solution
def theoretical_field(k, x, dx):
    k_plus = k + jnp.pi / dx
    k_min = k - jnp.pi / dx
    x = jnp.maximum(x, 1e-12)  # fixing x = 0

    # Calculating field
    first_term = 2 * 1j * jnp.pi + expi(1j * k_min * x) - expi(1j * k_plus * x)
    first_term *= dx / (4 * k * jnp.pi) * jnp.exp(-1j * k * x)

    second_term = expi(-1j * k_min * x) - expi(-1j * k_plus * x)
    second_term *= dx / (4 * k * jnp.pi) * jnp.exp(1j * k * x)

    field = first_term + second_term

    # Limit
    limit = dx * jnp.exp(1j * k * x) / (-2 * 1j * k)
    return field, limit


# %%
phi_a, limit = theoretical_field(k, x, dx)

# %%
plt.plot(x, jnp.abs(phi_a) ** 2, label="Theoretical")
plt.plot(x, jnp.abs(limit) ** 2, label="Limit")
plt.title("Intensity of field")
plt.legend()


# Done, now let's start working on the other method
shape = (1, 200)
x_sim = x[None, :]
k_sq = jnp.full(shape, k**2)

# Picking k0 and e
k0_sq = (jnp.min(jnp.real(k_sq)) + jnp.max(jnp.real(k_sq))) / 2


# %% Boundary values
def absorbing_boundary(x, k0, alpha, N):
    ax = alpha * x
    P = reduce(
        lambda P, n: P + (ax**n / factorial(n, exact=True)),
        range(N + 1),
        jnp.zeros_like(x),
    )
    boundary = (
        alpha**2
        * (N - ax + 2 * 1j * k0 * x)
        * ax ** (N - 1)
        / (P * factorial(N, exact=True))
    )
    return boundary


# %%
k0 = jnp.sqrt(k0_sq)
alpha = 1.0
x_bound = jnp.linspace(0, 5000, 2000) * dx / lambda_
boundary = absorbing_boundary(x_bound, k0, alpha, 4)
# %%
plt.plot(jnp.real(boundary), label="real component")
plt.axhline(-(alpha**2), color="k", linestyle="--")

plt.plot(jnp.imag(boundary), label="Imaginary component")
plt.axhline(2 * k0 * alpha, color="k", linestyle="--")
plt.title("Absorbing boundary value")
plt.legend()
plt.xlabel("Distance from boundary [wavelengths]")


# %% Getting real alpha and adding boundary values
alpha = jnp.sqrt(-2 * k0**2 + jnp.sqrt(4 * k0**4 + beta**2))
assert jnp.allclose(jnp.abs(-(alpha**2) + 2 * 1j * k0 * alpha), beta, atol=1e-3)

x_bound = dx * jnp.arange(1, int(boundary_thickness * lambda_ / dx) + 1)
k_sq_boundary = absorbing_boundary(x_bound, k0, alpha, 4)[None, :]
k_sq_padded = jnp.concatenate(
    [k_sq_boundary[:, ::-1], k_sq - k0_sq, k_sq_boundary], axis=1
)
# k_sq_padded = k_sq - k0_sq


# Setting epsilon, defining potential
epsilon = 40  # 1.1 * jnp.max(jnp.abs(k_sq_padded))  # no k0 as we already took it off
V = k_sq_padded - 1j * epsilon

## %% Prepping arrays
source = jnp.zeros(shape, dtype=jnp.complex64).at[:, 0].set(1.0)
source_padded = jnp.concatenate(
    [jnp.zeros_like(k_sq_boundary), source, jnp.zeros_like(k_sq_boundary)], axis=1
)

# Making greens array
ky = jnp.fft.fftfreq(V.shape[0], dx)
kx = jnp.fft.fftfreq(V.shape[1], dx)
k_grid = jnp.stack(jnp.meshgrid(ky, kx, indexing="ij"))
Gk = 1 / (4 * jnp.pi**2 * jnp.sum(k_grid**2, axis=0) - k0_sq - 1j * epsilon)


# %%
def update(field, _):
    error = field - jnp.fft.ifft2(Gk * jnp.fft.fft2(V * field + source_padded))
    correction = -1j / epsilon * V * error
    field = field + correction
    return field, correction


n_iterations = 10
initial = source_padded  # jnp.zeros_like(source_padded)#.at[:, 100:-100].set(phi_a)
field, correction = jax.lax.scan(update, initial, length=n_iterations)
field_n = field.squeeze()  # [100:-100]
# %%
plt.plot(jnp.abs(field_n) ** 2)
# plt.plot(jnp.abs(phi_a)**2, '--')
# plt.xlim([0, 50])
# %%
field_diff = (
    jax.lax.scan(update, initial, length=n_iterations + 1)[0]
    - jax.lax.scan(update, initial, length=n_iterations)[0]
)
# %%
