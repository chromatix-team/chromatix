from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import fresnel

import chromatix.functional as cf

D = 40.0
w = D / 10.0
z = 100.0
spectrum = 0.532
n = 1.33


def analytical_result_square_aperture(x, z, w, spectrum, n):
    Nf = (w / 2) ** 2 / (spectrum / n * z)

    def intensity(x):
        Smin, Cmin = fresnel(jnp.sqrt(2 * Nf) * (1 - 2 * x / w))
        Splus, Cplus = fresnel(jnp.sqrt(2 * Nf) * (1 + 2 * x / w))

        return 1 / jnp.sqrt(2) * (Cmin + Cplus) + 1j / jnp.sqrt(2) * (Smin + Splus)

    U = (
        jnp.exp(1j * 2 * jnp.pi * z * n / spectrum)
        / 1j
        * intensity(x[..., 0])
        * intensity(x[..., 1])
    )
    # Return U/l as the input field has area l^2
    return U / w


@pytest.mark.parametrize(
    ("shape", "pad_width"),
    [((256, 256), (512, 512)), ((1024, 256), (256, 512))],
)
def test_transform_propagation(shape, pad_width):
    dxi = D / np.array(shape)
    spacing = dxi[np.newaxis]
    field = cf.plane_wave(
        shape, spacing, (spectrum, 1.0), pupil=partial(cf.square_pupil, w=D)
    )
    out_field = cf.transform_propagate(field, z, n, pad_width=pad_width)
    I_numerical = out_field.intensity.squeeze()
    xi = np.array(out_field.grid.squeeze())
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(jnp.array(U_analytical)) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(I_analytical**2)
    assert rel_error < 2e-2
    field = cf.plane_wave(shape, spacing, (spectrum, 1.0))
    field = cf.square_pupil(field, w)  # Pupil after plane wave to lose some power
    out_field = cf.transform_propagate(field, z, n, pad_width=0)
    back_field = cf.transform_propagate(out_field, -z, n, pad_width=0)
    assert jnp.allclose(back_field.u, field.u, rtol=2e-5)


@pytest.mark.parametrize(
    ("shape"),
    [((256, 256)), ((1024, 256))],
)
def test_transform_sas_propagation(shape):
    dxi = D / np.array(shape)
    spacing = dxi[np.newaxis]
    field = cf.plane_wave(
        shape, spacing, (spectrum, 1.0), pupil=partial(cf.square_pupil, w=D)
    )
    out_field = cf.transform_propagate_sas(field, z, n)
    I_numerical = out_field.intensity.squeeze()
    xi = np.array(out_field.grid.squeeze())
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(I_analytical**2)
    assert rel_error < 2e-2


@pytest.mark.parametrize(
    ("shape", "pad_width"),
    [((256, 256), (512, 512)), ((1024, 256), (256, 512))],
)
def test_transfer_propagation(shape, pad_width):
    dxi = D / np.array(shape)
    spacing = dxi[np.newaxis]
    field = cf.plane_wave(
        shape, spacing, (spectrum, 1.0), pupil=partial(cf.square_pupil, w=D)
    )
    out_field = cf.transfer_propagate(field, z, n, pad_width=pad_width, mode="same")
    I_numerical = out_field.intensity.squeeze()
    xi = np.array(out_field.grid.squeeze())
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(I_analytical**2)
    assert rel_error < 2e-2


@pytest.mark.parametrize(
    ("shape", "pad_width"),
    [((256, 256), (512, 512)), ((1024, 256), (256, 512))],
)
def test_asm_without_evanescent_propagation(shape, pad_width):
    dxi = D / np.array(shape)
    spacing = dxi[np.newaxis]
    remove_evanescent = True
    bandlimit = False
    field = cf.plane_wave(
        shape, spacing, (spectrum, 1.0), pupil=partial(cf.square_pupil, w=D)
    )
    out_field = cf.asm_propagate(
        field,
        z,
        n,
        pad_width=pad_width,
        mode="same",
        remove_evanescent=remove_evanescent,
        bandlimit=bandlimit,
    )
    I_numerical = out_field.intensity.squeeze()
    # ASM without evanescent waves (exact transfer) is a bit worse here since it requires a lot of padding.
    # TODO: Find better test case.
    xi = np.array(out_field.grid.squeeze())
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(I_analytical**2)
    assert rel_error < 2e-2
    field = cf.plane_wave(shape, spacing, (spectrum, 1.0))
    field = cf.square_pupil(field, w)  # Pupil after plane wave to lose some power
    out_field = cf.asm_propagate(
        field,
        z,
        n,
        pad_width=0,
        mode="same",
        remove_evanescent=remove_evanescent,
        bandlimit=bandlimit,
    )
    back_field = cf.asm_propagate(
        out_field,
        -z,
        n,
        pad_width=0,
        mode="same",
        remove_evanescent=remove_evanescent,
        bandlimit=bandlimit,
    )
    # NOTE(dd/2025-09-29): This nondeterministically fails for shape (1024, 256)
    # on GPU, but works on CPU.
    assert jnp.allclose(back_field.u, field.u, rtol=2e-5)


@pytest.mark.parametrize(
    ("shape", "pad_width"),
    [((256, 256), (512, 512)), ((1024, 256), (256, 512))],
)
def test_asm_propagation(shape, pad_width):
    dxi = D / np.array(shape)
    spacing = dxi[np.newaxis]
    bandlimit = False
    field = cf.plane_wave(
        shape, spacing, (spectrum, 1.0), pupil=partial(cf.square_pupil, w=D)
    )
    out_field = cf.asm_propagate(
        field, z, n, pad_width=pad_width, mode="same", bandlimit=bandlimit
    )
    I_numerical = out_field.intensity.squeeze()
    xi = np.array(out_field.grid.squeeze())
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(I_analytical**2)
    assert rel_error < 2e-2


@pytest.mark.parametrize(
    ("shape", "pad_width"),
    [((256, 256), (512, 512)), ((1024, 256), (256, 512))],
)
def test_blas_propagation(shape, pad_width):
    dxi = D / np.array(shape)
    spacing = dxi[np.newaxis]
    bandlimit = True
    field = cf.plane_wave(
        shape, spacing, (spectrum, 1.0), pupil=partial(cf.square_pupil, w=D)
    )
    out_field = cf.asm_propagate(
        field, z, n, pad_width=pad_width, mode="same", bandlimit=bandlimit
    )
    I_numerical = out_field.intensity.squeeze()
    xi = np.array(out_field.grid.squeeze())
    U_analytical = analytical_result_square_aperture(xi, z, D, spectrum, n)
    I_analytical = jnp.abs(U_analytical) ** 2
    rel_error = jnp.mean((I_analytical - I_numerical) ** 2) / jnp.mean(I_analytical**2)
    assert rel_error < 2e-2


def test_transform_multiple():
    field_after_first_lens = cf.objective_point_source(
        (512, 512), 0.3, (spectrum, 1.0), 0, f=10.0, n=1.0, NA=0.8
    )
    field_after_first_propagation = cf.transform_propagate(
        field_after_first_lens, z=10.0, n=1, pad_width=256
    )
    field_after_second_propagation = cf.transform_propagate(
        field_after_first_propagation, z=10.0, n=1, pad_width=256
    )
    assert field_after_second_propagation.intensity.squeeze()[256, 256] != 0.0
