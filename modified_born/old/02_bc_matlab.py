# %% imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import expi

# %% Settings and sample

lambda_ = 1.0
dx = lambda_ / 4
beta = 0.2
boundary_thickness = 25
shape = (1, 200)
n_pad = 100

# %% Making sample
_n = 1.33
n_sample = jnp.full(shape, _n)
k = 2 * jnp.pi * _n / lambda_
e_r = n_sample**2
e_r_centre = 1 / 2 * (jnp.min(e_r) + jnp.max(e_r))


# %% Padding sample
e_r = jnp.pad(e_r, ((0, 0), (n_pad, n_pad)), mode="edge")
e_0 = jnp.mean(e_r)
k0 = jnp.sqrt(e_0) * 2 * jnp.pi / (lambda_ / dx)  # k0 in 1/pixels
c = beta * k0**2 / (2 * k0)


def f_boundary(r):
    numerator = c**6 * r**4.0 * (5.0 + (2.0 * 1j * k0 - c) * r)
    denominator = (
        120
        + 120 * c * r
        + 60 * c**2 * r**2
        + 20 * c**3 * r**3
        + 5 * c**4 * r**4
        + c**5 * r**5
    )

    return 1 / k0**2 * numerator / denominator


r = jnp.concatenate(
    [jnp.arange(100, 0, -1), jnp.zeros((200)), jnp.arange(1, 101)], axis=0
)
e_r += f_boundary(r)
# %% Now let's run the simulation

k00 = 2 * jnp.pi / lambda_
k = jnp.sqrt(e_r_centre) * k00
V = e_r * k00**2 - k**2
epsilon = jnp.maximum(jnp.max(jnp.abs(V)), 1e-3)
V = V - 1j * epsilon
# %%
source = jnp.zeros(shape, dtype=jnp.complex64).at[:, 0].set(1.0)
source = jnp.concatenate([jnp.zeros((1, 100)), source, jnp.zeros((1, 100))], axis=1)

# Making greens array
ky = jnp.fft.fftfreq(V.shape[0], dx)
kx = jnp.fft.fftfreq(V.shape[1], dx)
k_grid = jnp.stack(jnp.meshgrid(ky, kx, indexing="ij"))
Gk = 1 / (4 * jnp.pi**2 * jnp.sum(k_grid**2, axis=0) - k**2 - 1j * epsilon)


# %%
def update(field, _):
    error = field - jnp.fft.ifft2(Gk * jnp.fft.fft2(V * field + source))
    correction = -1j / epsilon * V * error
    field = field + correction
    return field, correction


n_iterations = 10000
field, correction = jax.lax.scan(update, source, length=n_iterations)
field = field.squeeze()[100:-100]

# %%
plt.plot(jnp.abs(field) ** 2)
# %%


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
x = jnp.arange(200) * dx
phi_a, limit = theoretical_field(k, x, dx)

# %%
plt.plot(jnp.abs(phi_a) ** 2)
plt.plot(jnp.abs(field) ** 2)
# %%
err = jnp.mean(jnp.abs(phi_a - field) ** 2) / jnp.mean(jnp.abs(phi_a) ** 2)
print(err)
# %%
