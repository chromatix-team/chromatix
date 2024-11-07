# description: example of propagation through calcite crystal
# author: bothg
# date: 2024/08/21

# %%
from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
from samples import Sample, Source
from solvers import maxwell_solver

import chromatix.functional as cf

# %% Properties of the birefringent crystal
n_o = 1.486  # ordinary / slow axis
n_e = 1.658  # extraordinary / fast axis

wavelength = 0.5  # 500nm wavelength in mum
theta_calcite = jnp.pi / 4  # 45deg angle rotation of calcite
spacing = wavelength / 8
boundary_thickness = 5.0
beam_diameter = 2.5



# %% Making the crystal and rotating it
# The fast axis is along z, the optical axis.
R = lambda theta: jnp.array(
    [
       
        [jnp.cos(theta), 0, jnp.sin(theta)],
        [0, 1, 0],
        [-jnp.sin(theta), 0, jnp.cos(theta)]
    ]
)
n_calcite = jnp.diag(jnp.array([n_e, n_e, n_o]))#jnp.array([[n_e, 0, 0], [0, n_e, 0], [0, 0, n_o]])
n_calcite = R(-theta_calcite) @ n_calcite @ R(theta_calcite)
print(n_calcite)


# %% Making the actual refractive index with air before and after
n_sample = jnp.zeros((512, 1, 256, 3, 3))
n_sample = n_sample.at[:96].set(jnp.eye(3))
n_sample = n_sample.at[-96:].set(jnp.eye(3))
n_sample = n_sample.at[96:-96].set(n_calcite)
# %%

sample = Sample.init(
    n_sample,
    spacing,
    wavelength,
    boundary_type="pbl",
    boundary_width=(boundary_thickness, None, None),
    boundary_strength=0.35,
)

# %%
# TODO: Add gaussian window
x = jnp.linspace(-128, 128, 256) * spacing
z = jnp.linspace(0, 512, 512) * spacing
window_x = jnp.exp(-0.5 * x**2 / ((beam_diameter / 2) ** 2))
window_z = jnp.exp(-.5 * (z - beam_diameter)**2 / wavelength**2)
window = (window_x[None, :] * window_z[:, None])[:, None, :, None]

source = Source(
    field=jnp.zeros((*n_sample.shape[:3], 3), dtype=jnp.complex64)
    .at[:]
    .set(window_x[:, None] * cf.linear(jnp.pi / 4)),
    wavelength=wavelength,
)

# %%
field, stats = maxwell_solver(source, sample, max_iter=500)
# %%
plt.semilogy(stats["error"][: stats["n_iterations"]])

# %%
plt.figure(figsize=(15, 25))
plt.subplot(311)
plt.imshow(jnp.rot90(jnp.abs(field[:, 0, :, 0]) ** 2), cmap="jet")  # %%
plt.title("E_z")
plt.xlabel("z")
plt.ylabel("x")
plt.colorbar()

plt.subplot(312)
plt.imshow(jnp.rot90(jnp.abs(field[:, 0, :, 1]) ** 2), cmap="jet")  # %%
plt.title("E_y")
plt.xlabel("z")
plt.ylabel("x")
plt.colorbar()

plt.subplot(313)
plt.imshow(jnp.rot90(jnp.abs(field[:, 0, :, 2]) ** 2), cmap="jet")  # %%
plt.title("E_x")
plt.xlabel("z")
plt.ylabel("x")
plt.colorbar()
# %%
plt.imshow(jnp.log10(jnp.rot90(jnp.sum(jnp.abs(field.squeeze()) ** 2, axis=-1))), cmap="jet")  # %%
# %%
plt.show()
# %%
