import numpy as np
import jax.numpy as jnp
import chromatix.functional as cf
from chromatix import ScalarField


def test_ff_lens():
    field_after_first_lens = cf.objective_point_source(
        (512, 512), 0.3, (0.532, 1.0), 0, f=10.0, n=1.0, NA=0.8
    )
    field_after_second_lens = cf.ff_lens(field_after_first_lens, f=10.0, n=1, NA=None)
    field_after_third_lens = cf.ff_lens(field_after_second_lens, f=10.0, n=1, NA=None)
    field_after_second_lens_back = cf.ff_lens(
        field_after_third_lens, f=10.0, n=1, NA=None, inverse=True
    )

    assert jnp.allclose(
        field_after_second_lens.intensity,
        field_after_second_lens_back.intensity,
        atol=1e-5,
    )
    assert field_after_third_lens.u.squeeze()[256, 256] != 0.0


def test_df_lens():
    field_after_first_lens = cf.objective_point_source(
        (512, 512), 0.3, (0.532, 1.0), 0, f=10.0, n=1.0, NA=0.8
    )
    field_after_second_lens = cf.df_lens(
        field_after_first_lens, d=8.0, f=10.0, n=1, NA=None
    )
    field_after_third_lens = cf.df_lens(
        field_after_second_lens, d=8.0, f=10.0, n=1, NA=None
    )
    field_after_second_lens_back = cf.df_lens(
        field_after_third_lens, d=8.0, f=10.0, n=1, NA=None, inverse=True
    )

    # We don't test the exact fields as their spacing is different
    assert jnp.allclose(
        field_after_second_lens.power, field_after_second_lens_back.power
    )
    assert field_after_third_lens.u.squeeze()[256, 256] != 0.0


def test_high_na_ff_lens():

    def _sim_illumination_pupil(N: int, dx: float, wavelength: float, radius: int) -> ScalarField:
        """
        Creates a pupil of two points separated by ``2 * radius`` samples which
        should result in a sinusoidal pattern at the image plane.
        """
        field = cf.plane_wave((N, N), dx, wavelength, power=None)
        u = np.zeros((N, N), dtype=complex)
        c = N // 2
        u[c, c - radius] = 1.0
        u[c, c + radius] = 1.0
        return field.replace(u=jnp.asarray(u))

    # NOTE(dd/2026-07-30): Testing that zooming is not affected by the shape of
    # the input field (GitHub issue #185).
    wavelength = 0.532
    n = 1.5
    NA = 0.8
    f = 50.0
    dx = 0.5
    radius = 20
    output_shape = (256, 256)
    output_dx = 0.02
    small = cf.high_na_tube_lens(
        _sim_illumination_pupil(128, dx, wavelength, radius),
        f,
        n,
        NA,
        output_shape=output_shape,
        output_dx=output_dx,
    )
    padded = cf.high_na_tube_lens(
        _sim_illumination_pupil(256, dx, wavelength, radius),
        f,
        n,
        NA,
        output_shape=output_shape,
        output_dx=output_dx,
    )
    small = small.intensity
    padded = padded.intensity
    small = small / small.max()
    padded = padded / padded.max()
    # NOTE(dd/2026-07-30): Defocus phase is still calculated with different
    # sampling depending on the shape of the input field, which creates some
    # small difference in the results.
    assert jnp.allclose(small, padded, atol=1e-2)
    # NOTE(dd/2026-07-30): Testing that zooming goes to the correct dx (should
    # get a sinusoid pattern with period ``wavelength * f / (n * d_rho)``).
    d_rho = 2 * radius * dx
    expected_period = wavelength * f / (n * d_rho)
    out = cf.high_na_tube_lens(
        _sim_illumination_pupil(128, dx, wavelength, radius),
        f, n, NA, output_shape=output_shape, output_dx=output_dx,
    )
    row = np.asarray(out.intensity.squeeze()[output_shape[0] // 2, :])
    row = row - row.mean()
    padded_length = 1 << 16
    spectrum = np.abs(np.fft.rfft(row, padded_length))
    k = spectrum[1:].argmax() + 1
    a, b, c = spectrum[k - 1], spectrum[k], spectrum[k + 1]
    k_hat = k + 0.5 * (a - c) / (a - 2 * b + c)
    measured_period = (padded_length / k_hat) * output_dx
    assert np.abs((measured_period / expected_period) - 1.0) < 0.02
