import jax.numpy as jnp
from scipy.special import fresnel
from chromatix import Field
import chromatix.functional as cf

D = 40
z = 100
spectrum = 0.532
n = 1.33
Nf = (D / 2) ** 2 / (spectrum / n * z)


def analytical_result_square_aperture(x, z, D, spectrum, n):
    Nf = (D / 2) ** 2 / (spectrum / n * z)

    Smin, Cmin = fresnel(jnp.sqrt(2 * Nf) * (1 - 2 * x / D))
    Splus, Cplus = fresnel(jnp.sqrt(2 * Nf) * (1 + 2 * x / D))

    Ix = 1 / jnp.sqrt(2) * (Cmin + Cplus) + 1j / jnp.sqrt(2) * (Smin + Splus)
    U = jnp.exp(1j * 2 * jnp.pi * z * n / spectrum) / 1j * Ix[:, None] * Ix[None, :]
    # Return U/l as the input field has area l^2
    return U / D


def test_transform_propagation():
    N = 512
    dxi = D / N
    Q = 5
    N_pad = Q * N

    # Input field
    field = cf.empty_field((N, N), dxi, 0.532, 1.0)
    field = cf.plane_wave(field, pupil=lambda field: cf.square_pupil(field, dxi * N))
    out_field = cf.transform_propagate(field, z, n, N_pad=N_pad)
    I_numerical = out_field.intensity.squeeze()
    xi = out_field.dx.squeeze() * jnp.arange(-N / 2, N / 2)

    # Analytical
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(
        I_analytical**2
    )
    assert rel_error < 1e-3


def test_transfer_propagation():
    N = 512
    dxi = D / N
    Q = 5
    N_pad = Q * N

    # Input field
    field = cf.empty_field((N, N), dxi, 0.532, 1.0)
    field = cf.plane_wave(field, pupil=lambda field: cf.square_pupil(field, dxi * N))
    out_field = cf.transfer_propagate(field, z, n, N_pad=N_pad, mode="same")
    I_numerical = out_field.intensity.squeeze()
    xi = out_field.dx.squeeze() * jnp.arange(-N / 2, N / 2)

    # Analytical
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(
        I_analytical**2
    )
    assert rel_error < 1e-3


def test_exact_propagation():
    N = 512
    dxi = D / N
    Q = 5
    N_pad = Q * N

    # Input field
    field = cf.empty_field((N, N), dxi, 0.532, 1.0)
    field = cf.plane_wave(field, pupil=lambda field: cf.square_pupil(field, dxi * N))
    out_field = cf.exact_propagate(field, z, n, N_pad=N_pad, mode="same")
    I_numerical = out_field.intensity.squeeze()
    xi = out_field.dx.squeeze() * jnp.arange(-N / 2, N / 2)

    # Analytical
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(
        I_analytical**2
    )
    # Exact is a bit worse here since it requires a lot of padding.
    # TODO: Find better test case.
    assert rel_error < 1e-2


def test_propagate():
    N = 512
    dxi = D / N
    Q = 5
    N_pad = Q * N

    # Input field
    field = cf.empty_field((N, N), dxi, 0.532, 1.0)
    field = cf.plane_wave(field, pupil=lambda field: cf.square_pupil(field, dxi * N))
    out_field = cf.propagate(field, z, n, method="transfer", mode="same")
    I_numerical = out_field.intensity.squeeze()
    xi = out_field.dx.squeeze() * jnp.arange(-N / 2, N / 2)

    # Analytical
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(
        I_analytical**2
    )
    assert rel_error < 1e-3
