# %%
import chromatix.functional as cf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chromatix.utils.fft import fft, ifft
from jax import Array


# depolarised wave
def PTFT(k, km: Array) -> Array:
    Q = jnp.zeros((3, 3, *k.shape[1:]))

    # Setting diagonal
    Q_diag = 1 - k**2 / km**2
    Q = Q.at[jnp.diag_indices(3)].set(Q_diag)

    # Calculating off-diagonal elements
    q_ij = lambda i, j: -k[i] * k[j] / km**2
    # Setting upper diagonal
    Q = Q.at[0, 1].set(q_ij(0, 1))
    Q = Q.at[0, 2].set(q_ij(0, 2))
    Q = Q.at[1, 2].set(q_ij(1, 2))

    # Setting lower diagonal, mirror symmetry
    Q = Q.at[1, 0].set(q_ij(0, 1))
    Q = Q.at[2, 0].set(q_ij(0, 2))
    Q = Q.at[2, 1].set(q_ij(1, 2))

    # We move the axes to the back, easier matmul
    return jnp.moveaxis(Q.squeeze(-1), (0, 1), (-2, -1))


def bmatvec(a, b):
    return jnp.matmul(a, b[..., None]).squeeze(-1)


def thick_sample_vector(
    field: cf.VectorField, scatter_potential: Array, dz: float, n: float
) -> cf.VectorField:
    def P_op(u: Array) -> Array:
        phase_factor = jnp.exp(1j * kz * dz)
        return ifft(bmatvec(Q, phase_factor * fft(u)))

    def Q_op(u: Array) -> Array:
        return ifft(bmatvec(Q, fft(u)))

    def H_op(u: Array) -> Array:
        phase_factor = -1j * dz / 2 * jnp.exp(1j * kz * dz) / kz
        return ifft(bmatvec(Q, phase_factor * fft(u)))

    def propagate_slice(u, potential_slice):
        scatter_field = bmatvec(potential_slice, Q_op(u))
        return P_op(u) + H_op(scatter_field), None

    # Calculating k vector and PTFT
    # We shift k to align in k-space so we dont need shift just like Q
    km = 2 * jnp.pi * n / field.spectrum
    k = jnp.fft.ifftshift(field.k_grid, axes=field.spatial_dims)
    kz = jnp.sqrt(km**2 - jnp.sum(k**2, axis=0))
    k = jnp.concatenate([k, kz[None, ...]], axis=0)
    Q = PTFT(k, km)

    u, _ = jax.lax.scan(propagate_slice, field.u, scatter_potential)
    return field.replace(u=u)


# %% Settings

n = 1.0  # background refractive index
dz = 1.0  # propagation distance
potential = jnp.ones((10, 1, 512, 512, 1, 3, 3))


# %% Define incoming field

field = cf.point_source(
    (512, 512),
    1.0,
    0.532,
    1.0,
    1.0,
    1.0,
    amplitude=cf.linear(1 / 2 * jnp.pi),
    scalar=False,
    pupil=lambda f: cf.square_pupil(f, 300),
)

field_sample = thick_sample_vector(field, potential, dz, n)

# %%
plt.imshow(field_sample.intensity.squeeze())

# %%
