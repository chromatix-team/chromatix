# In this notebook we redevelop the code to do proper padding and z - y - x ordering.
# We've already made the samples, just checking.
# %% Imports
from functools import reduce
from numbers import Number

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from scipy.ndimage import distance_transform_edt
from scipy.special import factorial
import numpy as np
from jax.typing import ArrayLike
import jax
from scipy.signal.windows import tukey

# %%
n_sample = jnp.full((1000, 1000), 1.0)
n_sample = n_sample.at[jnp.triu_indices(n=1000)].set(1.55)[::-1, None, :]

plt.imshow(jnp.rot90(n_sample[:, 0, :]))
plt.colorbar()
# %%
plt.title("Sample - Offset")
plt.imshow(jnp.rot90(n_sample[:, 0, :]))
plt.xlabel("z")
plt.ylabel("x")
plt.colorbar(label="Refractive index")


# %% Now adding the absorbing boundary conditions
# Finding new shapes and rois
def add_bc(
    permittivity: Array,
    width: tuple[float | None, ...],
    spacing: float,
    wavelength: float,
    strength: float | None = None,
    alpha: float | None = None,
    order: int = 4,
) -> tuple[Array, tuple[slice, ...]]:
    """
    width is in wavelengths. Use None for periodic BCs.
    """

    # Figuring out new size
    n_pad = tuple(0 if width_i is None else int(width_i / spacing) for width_i in width)
    roi = tuple(slice(n, n + size) for n, size in zip(n_pad, permittivity.shape))

    # Padding permittivity to new size
    # We repeat the mean value
    padding = [(0, 0) for _ in range(permittivity.ndim)]
    for idx, n in enumerate(n_pad):
        padding[idx] = (n, n)
    permittivity = jnp.pad(permittivity, padding, mode="edge")

    # Gathering constants
    k0 = 2 * jnp.pi / wavelength
    km = k0 * jnp.sqrt(jnp.mean(permittivity))
    match (strength, alpha):
        case (Number(), None):
            alpha = strength * km / 2
        case (None, Number()):
            pass
        case (None, None):
            raise ValueError("Need at least strength or alpha set.")
        case (Number(), Number()):
            raise ValueError("Can only set either strength or alpha, not both.")
        case _:
            raise ValueError("Everything's wrong.")

    # Defining distance from sample
    r = jnp.ones_like(permittivity).at[roi].set(0)
    r = distance_transform_edt(r, sampling=spacing)

    # Making boundary
    ar = alpha * r
    P = reduce(
        lambda P, n: P + (ar**n / factorial(n, exact=True)),
        range(order + 1),
        jnp.zeros_like(ar),
    )

    numerator = alpha**2 * (order - ar + 2 * 1j * km * r) * ar ** (order - 1)
    denominator = P * factorial(order, exact=True)
    boundary = 1 / k0**2 * numerator / denominator

    # Inside the ROI it's 0
    boundary = boundary.at[roi].set(0)

    return permittivity + boundary, roi


# %%
spacing = 0.1
wavelength = 1.0
width = (25 / wavelength, None, 25 / wavelength)
alpha_boundary = 0.35
order = 4

permittivity, roi = add_bc(
    n_sample**2, width, spacing, wavelength, alpha=alpha_boundary, order=order
)
print(f"Padded permittivity shape: {permittivity.shape}")

# %%
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Real part of padded permittivity")
plt.imshow(jnp.rot90(permittivity[:, 0, :].real))
plt.colorbar(fraction=0.046, pad=0.04)
plt.xlabel("z")
plt.ylabel("x")

plt.subplot(122)
plt.title("Imaginary part of padded permittivity")
plt.imshow(jnp.rot90(permittivity[:, 0, :].imag))
plt.colorbar(fraction=0.046, pad=0.04)
# %%
plt.title("Crosss section of permittivity")
plt.plot(permittivity[:, 0, 950].real, label="real")
plt.plot(permittivity[:, 0, 950].imag, label="imaginary")
plt.axvline(roi[0].start, linestyle="--", color="k")
plt.axvline(roi[0].stop, linestyle="--", color="k")
plt.legend()
# %% How well did we do?
# We know that the real part of the permittivity needs to go to km^2 - alpha*2, although we're far form infinity
# and the imaginary part at 2*km*alpha from th original born series paper.
k0 = 2 * jnp.pi / wavelength
print(
    f"Real permittivity should saturate at {-alpha_boundary**2:.2f}, observed {k0**2 * (permittivity[0, 0, 0].real - jnp.mean(permittivity.real[roi])):.2f}"
)
print(
    f"Imaginary permittivity should saturate at {2 * alpha_boundary * 2 * jnp.pi * jnp.sqrt(jnp.mean(permittivity[roi].real) / wavelength):.2f}, observed {k0**2 * (permittivity[0, 0, 0].imag - jnp.mean(permittivity.imag[roi])):.2f}"
)

# %% Now we can start solving
# The optimal alpha is given by:
# Note this is NOT the alpha of the boundary, rather the background wavenumber we subtract
alpha_real = (jnp.min(permittivity.real) + jnp.max(permittivity.real)) / 2
alpha_imag = jnp.max(jnp.abs(permittivity - alpha_real)) / 0.95
alpha = alpha_real + 1j * alpha_imag

print(f"Optimal background wavenumber: {alpha:.2f}")
print(f"Expected propagation speed [wavelength^-1]: {2 * alpha_real / alpha_imag :2f}")


# %% Padding shapes
# Because we have absorbing BCs, we don't need to double the size to prevent circular convolution, saving a lot of time!
# We just pad to the next power of 2:
def pad_fourier(x: Array) -> Array:
    # Pads to fourier friendly shapes (powers of 2), depending
    # on periodic or absorbing BCs
    def n_pad(size):
        new_size = int(2 ** (np.ceil(np.log2(size))))
        return new_size, (0, new_size - size)

    return zip(*[n_pad(shape) for shape in x.shape])


padded_shape, padding = pad_fourier(permittivity)
print(f"Padded shape will be {padded_shape}")


# %% Now making the k-grid and the greens function
k0 = 2 * jnp.pi / wavelength
ks = [2 * jnp.pi * jnp.fft.fftfreq(shape, spacing) for shape in padded_shape]
k_grid = jnp.stack(jnp.meshgrid(*ks, indexing="ij"), axis=-1)

# Making sure the k_grid is z -y - x ordered.
assert jnp.diff(k_grid[:, 0, 0, 0])[0] != 0.0
assert jnp.diff(k_grid[0, 0, :, 2])[0] != 0.0


# %% Making the Greens function
def G_fn(k: Array, k0: ArrayLike, alpha: Array) -> Array:
    k_sq = jnp.sum(k**2, axis=-1)[..., None, None]
    k_cross = k[..., :, None] * k[..., None, :] / (alpha * k0**2)
    return (jnp.eye(3) - k_cross) / (k_sq - alpha * k0**2)


G = G_fn(k_grid, k0, alpha)
print(f"Shape of Greens function {G.shape}")

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Real part of Gzz")
plt.imshow(jnp.fft.fftshift(G[:, 0, :, 0, 0].real))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(132)
plt.title("Imaginary part of Gxx")
plt.imshow(jnp.fft.fftshift(G[:, 0, :, 2, 2].imag))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(133)
plt.title("Real part of Gzx")
plt.imshow(jnp.fft.fftshift(G[:, 0, :, 0, 2].real))
plt.colorbar(fraction=0.046, pad=0.04)

# %% Making the potential
V = permittivity - alpha


# %% Now making a source. To prevent aliasing, we settle on a 2D tukey
def make_source(
    permittivity,
):
    # We first make a grid
    N_z, N_y, N_x = permittivity.shape
    z = spacing * (jnp.linspace(0, (N_z - 1), N_z) - N_z / 2)

    # Sinc options
    z_loc = 50
    width = wavelength / 4

    # Adding longitudinal apodisation
    source = jnp.sinc((z + z_loc) / width)

    alpha = 0.5
    width_x = int(20 / spacing)
    width_y = 1
    n_pad = (N_x - width_x) // 2
    mask = (
        tukey(width_y, alpha, sym=False)[:, None]
        * jnp.pad(tukey(width_x, alpha, sym=False), ((n_pad, n_pad)))[None, :]
    )

    source = (
        source[:, None, None, None]
        * mask[None, ..., None]
        * (k0**2 * jnp.array([0, 1, 1]))
    )
    return source


source = make_source(permittivity)

# %%
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title(f"Ex of source")
plt.imshow(jnp.rot90(source[:, 0, :, 2]))
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(122)
plt.title("Ey of source")
plt.imshow(jnp.rot90(source[:, 0, :, 1]))
plt.colorbar(fraction=0.046, pad=0.04)


# %% These function we use solve the problem
def bmatvec(mat: Array, vec: Array) -> Array:
    return jnp.matmul(mat, vec[..., None]).squeeze(-1)


def pad(field: Array) -> Array:
    return jnp.pad(field, (*padding, (0, 0)))


def crop(field: Array) -> Array:
    return field[: V.shape[0], : V.shape[1], : V.shape[2], :]


fft = lambda x: jnp.fft.fftn(x, axes=(0, 1, 2))
ifft = lambda x: jnp.fft.ifftn(x, axes=(0, 1, 2))


def propagate(G: Array, field: Array) -> Array:
    return crop(ifft(bmatvec(G, fft(pad(field)))))


def update_fn(args):
    field, history, iteration = args

    # New field
    dE = (
        1j
        / alpha_imag
        * V[..., None]
        * (propagate(G, k0**2 * V[..., None] * field + source) - field)
    )

    # Calculating change
    delta = jnp.mean(jnp.abs(dE) ** 2) / jnp.mean(jnp.abs(field) ** 2)

    return field + dE, history.at[iteration].set(delta), iteration + 1


def cond_fn(args) -> bool:
    _, history, iteration = args
    return (history[iteration - 1] > rtol) & (iteration < max_iter)


# %%
rtol = 1e-6
max_iter = 1000

init = update_fn((jnp.zeros_like(source), jnp.zeros(max_iter), 0))
field, history, iteration = jax.block_until_ready(
    jax.lax.while_loop(cond_fn, update_fn, init)
)
history = history[:iteration]
# %%
plt.title("Relative change in field")
plt.semilogy(history)
plt.ylabel("dE")
plt.xlabel("Iteration")

# %%
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.title("Ex")
plt.imshow(jnp.rot90(jnp.abs(field[roi][:, 0, :, 2])), vmin=0.0, vmax=1.2)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(132)
plt.title("Ey")
plt.imshow(jnp.rot90(jnp.abs(field[roi][:, 0, :, 1])), vmin=0.0, vmax=1.2)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(133)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(field[roi][:, 0, :, 0])), vmin=0.0, vmax=1.2)
plt.colorbar(fraction=0.046, pad=0.04)

# %% As a check - is there any energy in the padding area?
total_power = jnp.sum(jnp.abs(field) ** 2)
no_padding_power = jnp.sum(jnp.abs(field[:1500, :, :1500]) ** 2)
power_in_padding = total_power - no_padding_power
power_in_BC = no_padding_power - jnp.sum(jnp.abs(field[roi] ** 2))

print(f"Fraction of power in padding: {power_in_padding / total_power:.2f}")
print(f"Fraction of power in BC: {power_in_BC / total_power:.2f}")

# %% Plotting everything together:
plt.figure(figsize=(20, 5))
plt.subplot(141)
plt.title("Sample - rotated air - glass interface ")
plt.imshow(jnp.rot90(n_sample[:, 0, :]))
plt.xlabel("z")
plt.ylabel("x")
plt.colorbar(fraction=0.046, pad=0.04)


plt.subplot(142)
plt.title("Ex")
plt.imshow(jnp.rot90(jnp.abs(field[roi][:, 0, :, 2])), vmin=0.0, vmax=1.2)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(143)
plt.title("Ey")
plt.imshow(jnp.rot90(jnp.abs(field[roi][:, 0, :, 1])), vmin=0.0, vmax=1.2)
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(144)
plt.title("Ez")
plt.imshow(jnp.rot90(jnp.abs(field[roi][:, 0, :, 0])), vmin=0.0, vmax=1.2)
plt.colorbar(fraction=0.046, pad=0.04)


# %%
