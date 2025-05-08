import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def multi_paganin(
    data: Array,
    distances: Array,
    wavelength: float,
    voxelsize: float,
    delta_beta: float,
    eps: float = 1e-7,
) -> tuple[Array, Array]:
    """
    Phase retrieval based on the MultiPaganin method using JAX.

    Parameters
    ----------
    data : ndarray, float32
        Input data for several distances, shape (ndist, n, n)
    distances : ndarray
        Distances in meters, shape (ndist,)
    wavelength : float
        Wavelength in meters
    voxelsize : float
        Voxel size in meters
    delta_beta : float
        Ratio between the real and imaginary parts of the refractive index
    eps : float = 1e-7
        Constant to avoid division by zero

    Returns
    -------
    phase : ndarray
        Recovered phase of the object, shape (ntheta, n, n)
    """

    def update(
        prev: tuple[Array, Array], data_per_dist: tuple[Array, Array]
    ) -> tuple[tuple[Array, Array], None]:
        numerator, denominator = prev
        measurement, dist = data_per_dist

        rad_freq = jnp.fft.fft2(measurement)
        taylor_exp = 1.0 + wavelength * dist * jnp.pi * delta_beta * f_sq_grid

        return (numerator + taylor_exp * rad_freq, denominator + taylor_exp**2), None

    # Setup frequencies
    n = data.shape[-1]
    freqs = jnp.fft.fftfreq(n, d=voxelsize)
    f_sq_grid = jnp.sum(
        jnp.stack(jnp.meshgrid(freqs, freqs, indexing="xy")) ** 2, axis=0
    )

    (numerator, denominator), _ = jax.lax.scan(
        update,
        (
            jnp.zeros((n, n), dtype=jnp.complex64),
            jnp.full((n, n), eps, dtype=jnp.complex64),
        ),
        (data, distances),
    )

    dphi = (
        -1 / 2 * delta_beta * jnp.log(jnp.real(jnp.fft.ifft2(numerator / denominator)))
    )

    return dphi + 1j * (dphi - dphi.min()) / delta_beta

def register_integer_shift(image, reference):
    correlation = jax.scipy.signal.correlate(
        reference, image, mode="same", method="fft"
    )
    max_corr_idx = jnp.stack(
        jnp.unravel_index(jnp.argmax(correlation), correlation.shape)
    )
    shift = jnp.array(correlation.shape) // 2 - max_corr_idx
    return shift


def _upsampled_dft_jax(
    data: jnp.ndarray,
    ups: int,
    upsample_factor: float = 1,
    axis_offsets: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Compute an ups×ups DFT patch around the given offsets for 2D frequency data.

    Args:
        data: 2D array (H, W) of complex cross-power spectrum.
        ups: size of the upsampled patch (ups × ups).
        upsample_factor: fractional upsampling factor.
        axis_offsets: length-2 array [off_y, off_x] in the upsampled grid.

    Returns:
        2D array (ups, ups) of the localized DFT patch.
    """
    im2pi = 1j * 2 * jnp.pi
    f = jnp.arange(ups)

    # 1) partial DFT along width (x) direction
    freq_x = jnp.fft.fftfreq(
        data.shape[1], d=upsample_factor
    )  # shape (W,)                        # shape (ups,)
    kernel_x = jnp.exp(
        -im2pi * ((f - axis_offsets[1])[:, None] * freq_x[None, :])
    )  # (ups, W)
    tdata = jnp.einsum("pw,hw->ph", kernel_x, data)  # shape (ups, H)

    # 2) partial DFT along height (y) direction
    freq_y = jnp.fft.fftfreq(
        data.shape[0], d=upsample_factor
    )  # shape (H,)                                  # shape (ups,)
    kernel_y = jnp.exp(
        -im2pi * ((f - axis_offsets[0])[:, None] * freq_y[None, :])
    )  # (ups, H)
    # rec: sum over y
    rec = jnp.einsum("ph,qh->pq", tdata, kernel_y)  # shape (ups, ups)

    return rec


def register_shift(
    image: jnp.ndarray,
    reference: jnp.ndarray,
    upsample_factor: int | None = None,
):
    """
    Efficient sub-pixel image translation registration (JAX).

    Args:
      src_image: 2D array (H, W) to register.
      target_image: 2D array (H, W) reference.
      upsample_factor: upsampling for fractional accuracy.
      space: 'real' (FFT input) or 'fourier' (inputs already spectra).
    Returns:
      shifts: [dy, dx] required to align src to target.
    """

    shifts = register_integer_shift(image, reference)

    # 3) Sub-pixel refinement
    if upsample_factor is not None:
        image_product = jnp.fft.fft2(image) * jnp.fft.fft2(reference).conj()
        # coarse estimate on upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        ups = int(np.ceil(upsample_factor * 1.5))
        dftshift = np.fix(ups / 2.0)

        # compute offsets for DFT patch
        axis_offsets = dftshift - shifts * upsample_factor

        # localized DFT patch
        patch = _upsampled_dft_jax(
            jnp.conj(image_product), ups, upsample_factor, axis_offsets
        )

        # locate sub-pixel peak in patch
        max_corr_idx = jnp.stack(
            jnp.unravel_index(jnp.argmax(jnp.abs(patch.conj())), patch.shape)
        )
        maxima = max_corr_idx - dftshift

        # combine integer + fractional
        shifts = shifts + maxima[::-1] / upsample_factor

    return shifts
