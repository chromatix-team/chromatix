import jax
import jax.numpy as jnp
import numpy as np
from chromatix import Field
from jaxtyping import Array


def apply_shift(x: Array, shift: Array) -> Array:
    """Shift is in pixels."""
    n = x.shape[-1]
    x = jnp.pad(x, n // 2, "symmetric")
    f = jnp.fft.fftfreq(2 * n)
    f = jnp.stack(jnp.meshgrid(f, f, indexing="ij"), axis=-1)
    s = jnp.exp(-2 * jnp.pi * 1j * jnp.sum(f * shift, axis=-1))
    return jnp.fft.ifft2(s * jnp.fft.fft2(x))[n // 2 : n + n // 2, n // 2 : n + n // 2]


def shift_field(field: Field, shift: Array) -> Field:
    # TODO: move to field?
    return field.replace(u=apply_shift(field.u.squeeze(), shift)[None, ..., None, None])


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
