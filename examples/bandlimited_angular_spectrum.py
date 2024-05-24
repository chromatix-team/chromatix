"""
Example of "Band-Limited Angular Spectrum Method for Numerical Simulation 
of Free-Space Propagation in Far and Near Fields" (2010) by Matsushima and
Shimobaba.

Specifically trying to replicate Fig 9a from the paper for a rectangular
aperture.

TODO: implement numerical integration for comparison?
Something like this: https://github.com/ebezzam/waveprop/blob/a2d65116336bfb6e95732fd982e5c3ec2109cff3/waveprop/rs.py#L33

"""
from functools import partial
import numpy as np
import jax.numpy as jnp
from scipy.special import fresnel
import chromatix.functional as cf
import matplotlib.pyplot as plt


# setting like in BLAS paper (Fig 9) https://opg.optica.org/oe/fulltext.cfm?uri=oe-17-22-19662&id=186848
shape = (1024, 1024)
N_pad = (512, 512)
spectrum = 0.532    # wavelength in microns
dxi = 2 * spectrum
D = dxi * shape[0]    # field shape in microns
w = D / 2
z = 100 * D

dxi = D / np.array(shape)
spacing = dxi[..., np.newaxis]
n = 1  # refractive index of medium

# # setting like https://github.com/chromatix-team/chromatix/blob/7304cd312b28eebc2f15c3c466e53074141d553b/tests/test_propagate.py#L34C1-L52C28
# D = 40 # microns
# z = 100   # microns 
# spectrum = 0.532    # microns
# shape = (512, 512)
# N_pad = (512, 512)
# n = 1  # refractive index of medium
# dxi = D / np.array(shape)
# spacing = dxi[..., np.newaxis]
# w = dxi[1] * shape[1]  # width of aperture in microns

print("Field shape [um]: ", D)
print("Width of aperture [um]: ", w)
print("Propagation distance [um]: ", z)


def analytical_result_square_aperture(x, z, D, spectrum, n):
    # TODO: this uses Fresnel approximation
    Nf = (D / 2) ** 2 / (spectrum / n * z)

    def I(x):
        Smin, Cmin = fresnel(jnp.sqrt(2 * Nf) * (1 - 2 * x / D))
        Splus, Cplus = fresnel(jnp.sqrt(2 * Nf) * (1 + 2 * x / D))

        return 1 / jnp.sqrt(2) * (Cmin + Cplus) + 1j / jnp.sqrt(2) * (Smin + Splus)

    U = jnp.exp(1j * 2 * jnp.pi * z * n / spectrum) / 1j * I(x[0]) * I(x[1])
    # Return U/l as the input field has area l^2
    return U / D

# Input field
field = cf.plane_wave(
    shape=shape,
    dx=spacing, 
    spectrum=spectrum, 
    spectral_density=1.0, 
    pupil=partial(cf.square_pupil, w=w)
)

# # Fresnel
# out_field_fresnel = cf.transform_propagate(field, z, n, N_pad=N_pad)
# I_fresnel = out_field_fresnel.intensity.squeeze()

# # Analytical (Fresnel)
# xi = np.array(out_field_fresnel.grid.squeeze())
# U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
# I_analytical = jnp.abs(U_analytical) ** 2

# Angular spectrum
out_field_asm = cf.asm_propagate(field, z, n, N_pad=N_pad, mode="same")
I_asm = out_field_asm.intensity.squeeze()

# Angular spectrum (bandlimited)
out_field_blas = cf.asm_propagate(field, z, n, N_pad=N_pad, mode="same", bandlimit=True)
I_blas = out_field_blas.intensity.squeeze()

# Compare
# -- compute error
intensities = [
    ["Input", field.intensity.squeeze()],
    # ["Analytical (Fresnel)", I_analytical],
    # ["Fresnel", I_fresnel],
    ["ASM", I_asm],
    ["BLAS", I_blas],
]
# for approach, intensity in intensities[2:]:
#     rel_error = jnp.mean((I_analytical - intensity) ** 2) / jnp.mean(
#         I_analytical**2
#     )
#     print(f"{approach} error: ", rel_error)

# -- plot
fig, axs = plt.subplots(1, len(intensities), figsize=(15, 4))
axs[0].set_ylabel("y (microns)")
for ax, (title, intensity) in zip(axs, intensities):
    ax.imshow(intensity, cmap="gray", extent=[-D/2, D/2, -D/2, D/2])
    ax.set_title(title)
    ax.set_xlabel("x (microns)")

plot_fn = "propagation_comparison.png"
plt.savefig(plot_fn)
print(f"Saved plot to {plot_fn}")
