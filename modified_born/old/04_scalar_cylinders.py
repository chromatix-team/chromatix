# This script shows how to use the 3d helmholtz solver for the cylinder
# dataset
# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
from samples import Sample, cylinders
from solvers import helmholtz_solver

# %%
cylinder_locs = jnp.array(
    [[-44.5, -44.5], [44.5, -44.5], [44.5, 44.5], [20.6, -18.0], [-10.4, 18.1]]
)

cylinder_radius = 5.0
n_cylinder = 1.36
n_mat = 1.33
wavelength = 0.532
spacing = wavelength / 4
sim_size = 100
sim_shape = int(sim_size / spacing)

n_sample = cylinders(
    (sim_shape, 1, sim_shape),
    spacing,
    cylinder_locs,
    cylinder_radius,
    n_mat,
    n_cylinder,
    antialiasing=10,
).squeeze()

# %%
plt.imshow(jnp.rot90(n_sample))
plt.ylabel("x")
plt.xlabel("z")
plt.title("Cylinder mask")
plt.colorbar()
plt.show()  # %%

# %%
source = (
    jnp.zeros_like(n_sample).at[0, :].set((2 * jnp.pi * n_mat / wavelength) ** 2)
)  # unit source at entry
sample = Sample.init(n_sample, source, spacing, wavelength, (25, 25))


# %%
field, stats = helmholtz_solver(sample, rtol=0, max_iter=10000)
# %%
plt.imshow(jnp.abs(field) ** 2)
plt.colorbar()
# %%
