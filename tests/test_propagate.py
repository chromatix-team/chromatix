from functools import partial
import pytest
import numpy as np
import jax.numpy as jnp
from scipy.special import fresnel
from chromatix import ScalarField
import chromatix.functional as cf

D = 40
z = 100
spectrum = 0.532
n = 1.33
Nf = (D / 2) ** 2 / (spectrum / n * z)


def analytical_result_square_aperture(x, z, D, spectrum, n):
    Nf = (D / 2) ** 2 / (spectrum / n * z)

    def I(x):
        Smin, Cmin = fresnel(jnp.sqrt(2 * Nf) * (1 - 2 * x / D))
        Splus, Cplus = fresnel(jnp.sqrt(2 * Nf) * (1 + 2 * x / D))

        return 1 / jnp.sqrt(2) * (Cmin + Cplus) + 1j / jnp.sqrt(2) * (Smin + Splus)

    U = jnp.exp(1j * 2 * jnp.pi * z * n / spectrum) / 1j * I(x[0]) * I(x[1])
    # Return U/l as the input field has area l^2
    return U / D


@pytest.mark.parametrize(
    ("shape", "N_pad"),
    [((256, 256), (512, 512)), ((1024, 256), (256, 512))],
)
def test_transform_propagation(shape, N_pad):
    dxi = D / np.array(shape)
    spacing = dxi[..., np.newaxis]

    # Input field
    field = cf.plane_wave(
        shape, spacing, 0.532, 1.0, pupil=partial(cf.square_pupil, w=dxi[1] * shape[1])
    )
    out_field = cf.transform_propagate(field, z, n, N_pad=N_pad)
    I_numerical = out_field.intensity.squeeze()

    # Analytical
    xi = np.array(out_field.grid.squeeze())
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(
        I_analytical**2
    )
    assert rel_error < 2e-2


@pytest.mark.parametrize(
    ("shape", "N_pad"),
    [((256, 256), (512, 512)), ((1024, 256), (256, 512))],
)
def test_transfer_propagation(shape, N_pad):
    dxi = D / np.array(shape)
    spacing = dxi[..., np.newaxis]

    # Input field
    field = cf.plane_wave(
        shape, spacing, 0.532, 1.0, pupil=partial(cf.square_pupil, w=dxi[1] * shape[1])
    )
    out_field = cf.transfer_propagate(field, z, n, N_pad=N_pad, mode="same")
    I_numerical = out_field.intensity.squeeze()

    # Analytical
    xi = np.array(out_field.grid.squeeze())
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(
        I_analytical**2
    )
    assert rel_error < 2e-2


@pytest.mark.parametrize(
    ("shape", "N_pad"),
    [((256, 256), (512, 512)), ((1024, 256), (256, 512))],
)
def test_exact_propagation(shape, N_pad):
    dxi = D / np.array(shape)
    spacing = dxi[..., np.newaxis]

    # Input field
    field = cf.plane_wave(
        shape, spacing, 0.532, 1.0, pupil=partial(cf.square_pupil, w=dxi[1] * shape[1])
    )
    out_field = cf.exact_propagate(field, z, n, N_pad=N_pad, mode="same")
    I_numerical = out_field.intensity.squeeze()

    # Analytical
    xi = np.array(out_field.grid.squeeze())
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(
        I_analytical**2
    )
    # Exact is a bit worse here since it requires a lot of padding.
    # TODO: Find better test case.
    assert rel_error < 2e-2


def test_transform_multiple():
    field_after_first_lens = cf.objective_point_source(
        (512, 512), 0.3, 0.532, 1.0, 0, f=10.0, n=1.0, NA=0.8
    )
    field_after_first_propagation = cf.transform_propagate(
        field_after_first_lens, z=10.0, n=1, N_pad=256
    )
    field_after_second_propagation = cf.transform_propagate(
        field_after_first_propagation, z=10.0, n=1, N_pad=256
    )

    assert field_after_second_propagation.intensity.squeeze()[256, 256] != 0.0
