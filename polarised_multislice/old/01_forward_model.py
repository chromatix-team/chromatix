# %%
import chromatix.functional as cf
import jax
import jax.numpy as jnp
from chromatix.utils.fft import fft, ifft

# %% Settings

n = 1.0  # background refractive index
dz = 1.0  # propagation distance

# %% Define incoming field
field = cf.plane_wave(
    (512, 512),
    1.0,
    0.532,
    1.0,
    amplitude=cf.linear(1 / 2 * jnp.pi),
    scalar=False,
    pupil=lambda f: cf.square_pupil(f, 300),
)


# %% Single slice


# depolarised wave
def PTFT(k, km):
    Q = jnp.zeros((3, 3, *k.shape[1:]))

    # Setting diagonal
    Q_diag = 1 - k**2 / km**2
    Q = Q.at[jnp.diag_indices(3)].set(Q_diag)

    # Calculating off-diagonal elements
    q_ij = lambda i, j: -k[i] * k[j] / km**2
    q_01 = q_ij(0, 1)
    q_02 = q_ij(0, 2)
    q_12 = q_ij(1, 2)

    # Setting upper diagonal
    Q = Q.at[0, 1].set(q_01)
    Q = Q.at[0, 2].set(q_02)
    Q = Q.at[1, 2].set(q_12)

    # Setting lower diagonal, mirror symmetry
    Q = Q.at[1, 0].set(q_01)
    Q = Q.at[2, 0].set(q_02)
    Q = Q.at[2, 1].set(q_12)

    return Q


def P(field, dz, n):
    u_hat = fft(field.u, shift=True)

    # Adding phase factor
    km = 2 * jnp.pi * n / field.spectrum
    kz = jnp.sqrt(km**2 - jnp.sum(field.k_grid**2, axis=0))
    k = jnp.stack([*field.k_grid, kz], axis=0)
    phase_factor = jnp.exp(1j * kz * dz)
    u_hat *= phase_factor

    # Doing matmul with Q
    Q = PTFT(k, km)
    Q = jnp.moveaxis(Q.squeeze(-1), (0, 1), (-2, -1))
    u_new = jnp.matmul(Q, u_hat[..., None]).squeeze(-1)
    return field.replace(u=ifft(u_new, shift=True))


def Q(field, n):
    u_hat = fft(field.u, shift=True)

    # Adding phase factor
    km = 2 * jnp.pi * n / field.spectrum
    kz = jnp.sqrt(km**2 - jnp.sum(field.k_grid**2, axis=0))
    k = jnp.stack([*field.k_grid, kz], axis=0)

    # Doing matmul with Q
    Q = PTFT(k, km)
    Q = jnp.moveaxis(Q.squeeze(-1), (0, 1), (-2, -1))
    u_new = jnp.matmul(Q, u_hat[..., None]).squeeze(-1)
    return field.replace(u=ifft(u_new, shift=True))


def H(field, dz, n):
    u_hat = fft(field.u, shift=True)

    # Adding phase factor
    km = 2 * jnp.pi * n / field.spectrum
    kz = jnp.sqrt(km**2 - jnp.sum(field.k_grid**2, axis=0))
    k = jnp.stack([*field.k_grid, kz], axis=0)
    phase_factor = -1j * dz / 2 * jnp.exp(1j * kz * dz) / kz
    u_hat *= phase_factor

    # Doing matmul with Q
    Q = PTFT(k, km)
    Q = jnp.moveaxis(Q.squeeze(-1), (0, 1), (-2, -1))
    u_new = jnp.matmul(Q, u_hat[..., None]).squeeze(-1)
    return field.replace(u=ifft(u_new, shift=True))


def propagate_slice(field, scatter_potential, dz, n):
    scatter_field = jnp.matmul(scatter_potential, Q(field, n=n)[..., None]).squeeze(-1)
    return P(field, dz, n) + H(scatter_field, dz, n)


def thick_sample_vector(field, scatter_potential, dz, n):
    def propagate(field, potential):
        return propagate_slice(field, potential, dz, n), None

    field, _ = jax.lax.scan(propagate, field, scatter_potential)
    return field


#
# Iterating over slices
# %%

# %%
