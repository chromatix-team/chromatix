# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import expi, factorial

# %%
wavelength = 1.0
spacing = wavelength / 4
N_bound = 4
boundary_max = 0.2

# %% Making sample
size = 200
x = jnp.arange(0, size) * spacing
k = 2 * jnp.pi / wavelength

# %% Analytical solution
kmin = k - jnp.pi / spacing
kplus = k + jnp.pi / spacing

solution = (
    spacing
    / (4 * k * jnp.pi)
    * jnp.exp(-1j * k * x)
    * (2 * 1j * jnp.pi + expi(1j * kmin * x) - expi(1j * kplus * x))
)
solution += (
    spacing
    / (4 * k * jnp.pi)
    * jnp.exp(1j * k * x)
    * (expi(-1j * kmin * x) - expi(-1j * kplus * x))
)

# %% sample making
kr = jnp.full(size, k)
s = jnp.zeros(size).at[0].set(1.0)  # source

# %% Calculating k0, epsilon and potential
k0 = jnp.sqrt((jnp.min(kr**2) + jnp.max(kr**2)) / 2)
epsilon = jnp.max(jnp.abs(kr**2 - k0**2)) + 3.0
v = kr**2 - k0**2 - 1j * epsilon

# %% Adding absorbing boundary
padding = int(25 * wavelength / spacing)
alpha = 1 / 2 * boundary_max

padded_grid = (
    jnp.arange(-(size[1] // 2 + padding), (size[1] // 2 + padding)) + 0.5
) * spacing
d = jnp.where(
    jnp.abs(padded_grid) - size[1] * spacing / 2 < 0,
    0,
    jnp.abs(padded_grid) - size[1] * spacing / 2,
)


def Pn(x, alpha, N):
    p = jnp.zeros_like(x)
    for i in range(N + 1):
        p += (alpha * x) ** i / factorial(i, exact=True)
    return p


absorbing = (
    alpha**2 * (N_bound - alpha * d + 2 * 1j * k0 * d) * (alpha * d) ** (N_bound - 1)
) / (Pn(d, alpha, N_bound) * factorial(N_bound, exact=True))

v_padded = jnp.pad(v, ((0, 0), (padding, padding)))
absorbing_v = v_padded + absorbing
# absorbing_v = jnp.pad(absorbing_v, ((0, 0), (56, 56)))
s_padded = jnp.pad(s, ((0, 0), (100, 100)))


# %% Making greens function
u = jnp.zeros_like(s)
kx = jnp.fft.fftfreq(v.size, spacing)
g = 1 / (kx**2 - k0**2 - 1j * epsilon)


# %%
def update(u):
    return u + 1j / epsilon * v * (jnp.fft.ifft(g * jnp.fft.fft(v * u + s)) - u)


# %%
for _ in range(1000):
    u = update(u)

# %%
plt.imshow(jnp.log(jnp.abs(u) ** 2))
# %%
plt.imshow(jnp.abs(u) ** 2)
# %%
plt.plot(jnp.log(jnp.abs(u[100, :]) ** 2))
plt.plot(kr[100, :])
# %%
