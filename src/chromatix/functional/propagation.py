from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from chromatix.field import crop, pad
from chromatix.functional.convenience import optical_fft
from chromatix.typing import ArrayLike, NumberLike
from chromatix.utils.fft import fft, ifft

from ..field import Field
from ..utils import _broadcast_1d_to_grid, _broadcast_1d_to_innermost_batch, l2_sq_norm

__all__ = [
    "transform_propagate",
    "compute_sas_precompensation",
    "transform_propagate_sas",
    "transfer_propagate",
    "exact_propagate",
    "asm_propagate",
    "kernel_propagate",
    "compute_transfer_propagator",
    "compute_exact_propagator",
    "compute_asm_propagator",
    "compute_padding_transform",
    "compute_padding_transfer",
    "compute_padding_exact",
]


def transform_propagate(
    field: Field,
    z: NumberLike,
    n: NumberLike,
    N_pad: int | tuple[int, int],
    cval: float = 0,
    skip_initial_phase: bool = False,
    skip_final_phase: bool = False,
) -> Field:
    """
    Fresnel propagate ``field`` for a distance ``z`` using transform method.
    This method is also called the single-FFT (SFT-FR) Fresnel propagation
    method. Note that this method changes the sampling of the resulting field.
    If the distance is negative, the field is propagated back to the source
    inverting essentially performing an inverse.

    Args:
        field: ``Field`` to be propagated.
        z: Distance to propagate.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for
            the propagation FFT. Use padding calculator utilities from
            ``chromatix.functional.propagation`` to compute the padding.
            !!! warning
                The pad value hould not be a Jax array, otherwise a
                ConcretizationError will arise when traced!
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        skip_initial_phase: Whether to skip the input phase change (before
            Fourier transforming). Defaults to False, in which case the input
            phase change is not skipped.
        skip_final_phase: Whether to skip the output phase change (after Fourier
            transforming). Defaults to False, in which case the output phase
            change is not skipped.
    """
    field = pad(field, N_pad, cval=cval)
    # Fourier normalization factor
    L_sq = field.spectrum * z / n
    # New field is optical_fft minus -1j factor
    if not skip_initial_phase:
        # Calculating input phase change (defining Q1)
        input_phase = (jnp.pi / L_sq) * l2_sq_norm(field.grid)
        field = field * jnp.exp(1j * input_phase)
    field = 1j * optical_fft(field, z, n)
    # Calculating output phase change (defining Q2)
    if not skip_final_phase:
        output_phase = (jnp.pi / L_sq) * l2_sq_norm(field.grid)
        field = field * jnp.exp(1j * output_phase)
    return crop(field, N_pad)


def compute_sas_precompensation(
    field: Field,
    z: NumberLike,
    n: NumberLike,
) -> Array:
    sz = np.array(field.spatial_shape)
    kz = 2 * z * jnp.pi * n / field.spectrum
    s = field.spectrum * field.k_grid / n
    s_sq = s**2
    N = _broadcast_1d_to_grid(sz, field.ndim)
    pad_factor = 2
    L = pad_factor * N * field.dx
    t = L / pad_factor / jnp.abs(z) + jnp.abs(s)
    W = jnp.prod((s_sq * (2 + 1 / t**2) <= 1), axis=0)
    H_AS = jnp.sqrt(
        jnp.maximum(0, 1 - jnp.sum(s_sq, axis=0))
    )  # NOTE(rh): or cast to complex? Can W be larger than the free-space limit?
    H_Fr = 1 - jnp.sum(s_sq, axis=0) / 2
    delta_H = W * jnp.exp(1j * kz * (H_AS - H_Fr))
    delta_H = jnp.fft.ifftshift(delta_H, axes=field.spatial_dims)
    return delta_H


def transform_propagate_sas(
    field: Field,
    z: NumberLike,
    n: NumberLike,
    cval: float = 0,
    skip_initial_phase: bool = False,
    skip_final_phase: bool = False,
) -> Field:
    """
    Propagate ``field`` for a distance ``z`` using the scalable angular spectrum
    (SAS) method. See https://doi.org/10.1364/OPTICA.497809 It changes the
    pixelsize like the transform method, but it is more accurate because it
    precompensates the phase error. Since it uses three FFTS, it is slower
    than the transform method. Note that the field is automatically padded by a
    factor of 2, so the pixelsize is halved.

    Note also that a negative propagation distance causes the code to apply
    the inverse propagation, i.e. propagating by a positive ``z`` and then
    a negative ``z`` would propagate you back to the original ``field``. In
    the negative ``z`` case the order of single step Fresnel propagation and
    precompensation is reversed.

    Args:
        field: ``Field`` to be propagated.
        z: Distance to propagate.
        n: A float that defines the refractive index of the medium.
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        skip_initial_phase: Whether to skip the input phase change (before
            Fourier transforming). Defaults to False, in which case the input
            phase change is not skipped.
        skip_final_phase: Whether to skip the output phase change (after Fourier
            transforming). Defaults to False, in which case the output phase
            change is not skipped.
    """
    # Don't change this pad_factor, only 2 is supported
    pad_factor = 2
    sz = jnp.array(field.spatial_shape)
    N_pad = sz // pad_factor
    field = pad(field, N_pad, cval=cval)

    def _forward(field: Field, z) -> Field:
        delta_H = compute_sas_precompensation(field, z, n)
        field = kernel_propagate(field, delta_H)
        field = transform_propagate(
            field, z, n, 0, 0, skip_initial_phase, skip_final_phase
        )
        return field

    def _inverse(field: Field, z) -> Field:
        field = transform_propagate(
            field, z, n, 0, 0, skip_initial_phase, skip_final_phase
        )
        delta_H = compute_sas_precompensation(field, z, n)
        field = kernel_propagate(field, delta_H)
        return field

    u = jax.lax.cond(z >= 0, _forward, _inverse, field, z)
    field = field.replace(u=u)
    return crop(field, N_pad)


def transfer_propagate(
    field: Field,
    z: NumberLike,
    n: NumberLike,
    N_pad: int,
    cval: float = 0,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Fresnel propagate ``field`` for a distance ``z`` using transfer method. This
    method is also called the convolutional Fresnel propagation (CV-FR) method.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for
            the propagation FFT. Use padding calculator utilities from
            ``chromatix.functional.propagation`` to compute the padding.
            !!! warning
                The pad value hould not be a Jax array, otherwise a
                ConcretizationError will arise when traced!
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    field = pad(field, N_pad, cval=cval)
    propagator = compute_transfer_propagator(field, z, n, kykx)
    field = kernel_propagate(field, propagator)
    if mode == "same":
        field = crop(field, N_pad)
    return field


def exact_propagate(
    field: Field,
    z: NumberLike,
    n: NumberLike,
    N_pad: int,
    cval: float = 0,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Propagate ``field`` for a distance ``z`` using exact transfer method.

    This method removes evanescent waves.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for
            the propagation FFT. Use padding calculator utilities from
            ``chromatix.functional.propagation`` to compute the padding.
            !!! warning
                The pad value hould not be a Jax array, otherwise a
                ConcretizationError will arise when traced!
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    field = pad(field, N_pad, cval=cval)
    propagator = compute_exact_propagator(field, z, n, kykx)
    field = kernel_propagate(field, propagator)
    if mode == "same":
        field = crop(field, N_pad)
    return field


def asm_propagate(
    field: Field,
    z: NumberLike,
    n: NumberLike,
    N_pad: int,
    cval: float = 0,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    bandlimit: bool = False,
    shift_yx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Propagate ``field`` for a distance ``z`` using angular spectrum method.

    This method does not remove evanescent waves.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        N_pad: A keyword argument integer defining the pad length for
            the propagation FFT. Use padding calculator utilities from
            ``chromatix.functional.propagation`` to compute the padding.
            !!! warning
                The pad value hould not be a Jax array, otherwise a
                ConcretizationError will arise when traced!
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format `[ky, kx]`.
        bandlimit: If ``True``, bandlimited the kernel according to "Band-
            Limited Angular Spectrum Method for Numerical Simulation of Free-
            Space Propagation in Far and Near Fields" (2009) by Matsushima and
            Shimobaba. Defaults to ``False``.
        shift_yx: If provided, defines a shift in microns in the destination
            plane. Should be an array of shape `[2,]` in the format `[y, x]`.
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    field = pad(field, N_pad, cval=cval)
    propagator = compute_asm_propagator(field, z, n, kykx, bandlimit, shift_yx)
    field = kernel_propagate(field, propagator)
    if mode == "same":
        field = crop(field, N_pad)
    return field


def kernel_propagate(field: Field, propagator: ArrayLike) -> Field:
    """
    Propagate an incoming ``Field`` by the given propagation kernel
    (``propagator``). This amounts to performing a Fourier convolution of the
    ``field`` and the ``propagator``.
    """
    axes = field.spatial_dims
    u = ifft(fft(field.u, axes=axes) * propagator, axes=axes)
    return field.replace(u=u)


def compute_transfer_propagator(
    field: Field,
    z: NumberLike,
    n: NumberLike,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
) -> Array:
    """
    Compute propagation kernel for Fresnel propagation.
    Returns an array that can be multiplied with the Fourier transform of the
    incoming Field, as performed by kernel_propagate.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or a 1D array.
        n: A float that defines the refractive index of the medium.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format `[ky, kx]`.
    """
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    L = jnp.sqrt(jnp.complex64(field.spectrum * z / n))  # lengthscale L
    phase = -jnp.pi * jnp.abs(L) ** 2 * l2_sq_norm(field.k_grid - kykx)
    return jnp.fft.ifftshift(jnp.exp(1j * phase), axes=field.spatial_dims)


def compute_exact_propagator(
    field: Field,
    z: NumberLike,
    n: NumberLike,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
) -> Array:
    """
    Compute propagation kernel for propagation with no Fresnel approximation.

    This version of the propagation kernel removes evanescent waves. Returns
    an array that can be multiplied with the Fourier transform of the incoming
    Field, as performed by kernel_propagate.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or an array of shape (Z 1
            1 1).
        n: A float that defines the refractive index of the medium.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format `[ky, kx]`.
    """
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    kernel = 1 - (field.spectrum / n) ** 2 * l2_sq_norm(field.k_grid - kykx)
    kernel = jnp.maximum(kernel, 0.0)  # removing evanescent waves
    phase = 2 * jnp.pi * (jnp.abs(z) * n / field.spectrum) * jnp.sqrt(kernel)
    kernel_field = jnp.where(z >= 0, jnp.exp(1j * phase), jnp.conj(jnp.exp(1j * phase)))
    return jnp.fft.ifftshift(kernel_field, axes=field.spatial_dims)


def compute_asm_propagator(
    field: Field,
    z: NumberLike,
    n: NumberLike,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    bandlimit: bool = False,
    shift_yx: ArrayLike | tuple[float, float] = (0.0, 0.0),
) -> Array:
    """
    Compute propagation kernel for propagation with no Fresnel approximation.

    This version of the propagation kernel does not remove evanescent waves,
    as per the definition of the angular spectrum method. Returns an array
    that can be multiplied with the Fourier transform of the incoming Field, as
    performed by kernel_propagate.

    Args:
        field: ``Field`` to be propagated.
        z: Distance(s) to propagate, either a float or an array of shape (Z 1
            1 1).
        n: A float that defines the refractive index of the medium.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format `[ky, kx]`.
        bandlimit: If ``True``, bandlimited the kernel according to "Band-
            Limited Angular Spectrum Method for Numerical Simulation of Free-
            Space Propagation in Far and Near Fields" (2009) by Matsushima and
            Shimobaba. Defaults to ``False``.
        shift_yx: If provided, defines a shift in microns in the destination
            plane. Should be an array of shape `[2,]` in the format `[y, x]`.
    """
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    kernel = 1 - (field.spectrum / n) ** 2 * l2_sq_norm(field.k_grid - kykx)
    delay = jnp.sqrt(jnp.complex64(kernel))  # keep evanescent modes
    # shift in output plane
    shift_yx = _broadcast_1d_to_grid(shift_yx, field.ndim)
    out_shift = 2 * jnp.pi * jnp.sum(field.k_grid * shift_yx, axis=0)
    # compute field
    phase = 2 * jnp.pi * (jnp.abs(z) * n / field.spectrum) * delay + out_shift
    kernel_field = jnp.where(z >= 0, jnp.exp(1j * phase), jnp.conj(jnp.exp(1j * phase)))
    if bandlimit:
        # Table 1 of "Shifted angular spectrum method for off-axis numerical
        # propagation" (2010) by Matsushima in vectorized form
        k_limit_p = ((shift_yx + 1 / (2 * field.dk)) ** (-2) * z**2 + 1) ** (
            -1 / 2
        ) / field.spectrum
        k_limit_n = ((shift_yx - 1 / (2 * field.dk)) ** (-2) * z**2 + 1) ** (
            -1 / 2
        ) / field.spectrum
        k0 = (1 / 2) * (
            jnp.sign(shift_yx + field.surface_area) * k_limit_p
            + jnp.sign(shift_yx - field.surface_area) * k_limit_n
        )
        k_width = (
            jnp.sign(shift_yx + field.surface_area) * k_limit_p
            - jnp.sign(shift_yx - field.surface_area) * k_limit_n
        )
        k_max = k_width / 2
        # obtain rect filter to bandlimit (Eq. 23)
        H_filter_yx = jnp.abs(field.k_grid - k0) <= k_max
        H_filter = H_filter_yx[0] * H_filter_yx[1]
        # apply filter
        kernel_field = kernel_field * H_filter
    return jnp.fft.ifftshift(kernel_field, axes=field.spatial_dims)


def compute_padding_transform(height: int, spectrum: float, dx: float, z: float) -> int:
    """
    Automatically compute the padding required for transform propagation.

    Args:
        height: Height of the field
        spectrum: spectrum of the field
        dx: spacing of the field
        z: A float that defines the distance to propagate.
    """
    # TODO: works only for square fields
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (spectrum * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2
    N = (np.ceil((Q * M) / 2) * 2).astype(int)
    N_pad = (N - M).astype(int)
    return N_pad


def compute_padding_transfer(height: int, spectrum: float, dx: float, z: float) -> int:
    """
    Automatically compute the padding required for transfer propagation.

    Args:
        height: Height of the field
        spectrum: spectrum of the field
        dx: spacing of the field
        z: A float that defines the distance to propagate.
    """
    # TODO: works only for square fields
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (spectrum * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2
    N = (np.ceil((Q * M) / 2) * 2).astype(int)
    N_pad = (N - M).astype(int)
    return N_pad


def compute_padding_exact(height: int, spectrum: float, dx: float, z: float) -> int:
    """
    Automatically compute the padding required for exact propagation.

    Args:
        height: Height of the field
        spectrum: spectrum of the field
        dx: spacing of the field
        z: A float that defines the distance to propagate.
    """
    # TODO: works only for square fields
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (spectrum * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2
    scale = np.max((spectrum / (2 * dx)))
    # assert scale < 1, "Can't do exact transfer when field.dx < lambda / 2"
    Q = Q / np.sqrt(1 - scale**2)  # minimum pad ratio for exact transfer
    N = (np.ceil((Q * M) / 2) * 2).astype(int)
    N_pad = (N - M).astype(int)
    return N_pad
