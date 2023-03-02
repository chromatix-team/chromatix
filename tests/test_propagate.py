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
    N = 256
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
        I_analytical ** 2
    )
    assert rel_error < 1e-3


def test_transfer_propagation():
    N = 256
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
        I_analytical ** 2
    )
    assert rel_error < 2e-3


def test_exact_propagation():
    N = 256
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
        I_analytical ** 2
    )
    # Exact is a bit worse here since it requires a lot of padding.
    # TODO: Find better test case.
    assert rel_error < 1e-2


def test_transform_multiple():
    empty_field = cf.empty_field((512, 512), 0.3, 0.532, 1.0)
    field_after_first_lens = cf.objective_point_source(
        empty_field, 0, f=10.0, n=1.0, NA=0.8
    )
    field_after_first_propagation = cf.transform_propagate(
        field_after_first_lens, z=10.0, n=1, N_pad=256
    )
    field_after_second_propagation = cf.transform_propagate(
        field_after_first_propagation, z=10.0, n=1, N_pad=256
    )

    assert field_after_second_propagation.intensity.squeeze()[256, 256] != 0.0
