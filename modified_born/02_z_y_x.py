# In this notebook we redevelop the code to do proper padding and z - y - x ordering.
# We've already made the samples, just checking.
# %% Imports
from functools import reduce
from numbers import Number

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from samples import vacuum_cylinders
from scipy.ndimage import distance_transform_edt
from scipy.special import factorial

# %%
n_sample = vacuum_cylinders()
print(f"Sample has shape {n_sample.shape}")

# %%
plt.title("Sample - cylinders in vacuum")
plt.imshow(jnp.rot90(n_sample[:, 0, :]))
plt.xlabel("z")
plt.ylabel("x")
plt.colorbar(label="Refractive index")

# %% Now adding the absorbing boundary conditions
# Finding new shapes and rois
def add_bc(permittivity: Array, width: tuple[float | None, ...], spacing: float, wavelength: float, strength: float | None = None, alpha: float | None = None, order: int = 4) -> tuple[Array, tuple[slice, ...]]:
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
    permittivity = jnp.pad(permittivity, padding, mode="constant", constant_values = jnp.mean(permittivity))

    # Gathering constants
    km = 2 * jnp.pi * jnp.sqrt(jnp.mean(permittivity)) / wavelength 
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
    boundary =  numerator / denominator
    
    # Inside the ROI it's 0
    boundary = boundary.at[roi].set(0)
    return permittivity + boundary, roi

# %%
spacing = 0.1 
wavelength = 1.0 
width = (25 / wavelength, None, 25 / wavelength)
alpha = 0.35 
order = 4

permittivity, roi = add_bc(n_sample**2, width, spacing, wavelength, alpha=alpha, order=order)
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
print(f"Real permittivity should saturate at {-alpha**2:.2f}, observed {permittivity[0, 0, 0].real - jnp.mean(permittivity.real[roi]):.2f}")
print(f"Real permittivity should saturate at {2 * alpha * 2 * jnp.pi * jnp.sqrt(jnp.mean(permittivity[roi].real) / wavelength):.2f}, observed {permittivity[0, 0, 0].imag - jnp.mean(permittivity.imag[roi]):.2f}")

# %%
