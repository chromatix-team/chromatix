from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import fftconvolve
from jaxtyping import Array, ArrayLike, Float, ScalarLike

from chromatix import Field, crop, pad, shift_grid
from chromatix.functional.convenience import optical_fft
from chromatix.functional.pupils import super_gaussian_pupil, tukey_pupil
from chromatix.typing import z
from chromatix.utils import (
    _broadcast_1d_to_innermost_batch,
    l2_sq_norm,
)
from chromatix.utils.czt import czt

__all__ = [
    "transform_propagate",
    "compute_sas_precompensation",
    "transform_propagate_sas",
    "transfer_propagate",
    "asm_propagate",
    "kernel_propagate",
    "compute_transfer_propagator",
    "compute_asm_propagator",
    "compute_padding_transform",
    "compute_padding_transfer",
    "compute_padding_exact",
]


def transform_propagate(
    field: Field,
    z: ScalarLike,
    n: ScalarLike,
    pad_width: int | tuple[int, int],
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
        z: How far to propagate as a scalar value in units of distance.
        n: A float that defines the (isotropic) refractive index of the medium.
        pad_width: A keyword argument integer defining the pad length for
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
    field = pad(field, pad_width, cval=cval)
    # Fourier normalization factor
    L_sq = field.broadcasted_wavelength * z / n
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
    return crop(field, pad_width)


def compute_sas_precompensation(
    field: Field,
    z: ScalarLike,
    n: ScalarLike,
) -> Array:
    kz = 2 * z * jnp.pi * n / field.broadcasted_wavelength
    s = field.broadcasted_wavelength * field.f_grid / n
    s_sq = s**2
    pad_factor = 2
    L = pad_factor * field.extent
    t = L / pad_factor / jnp.abs(z) + jnp.abs(s)
    W = jnp.prod((s_sq * (2 + 1 / t**2) <= 1), axis=-1)
    H_AS = jnp.sqrt(
        jnp.maximum(0, 1 - jnp.sum(s_sq, axis=-1))
    )  # NOTE(rh): Or cast to complex? Can W be larger than the free-space limit?
    H_Fr = 1 - jnp.sum(s_sq, axis=-1) / 2
    delta_H = W * jnp.exp(1j * kz * (H_AS - H_Fr))
    delta_H = jnp.fft.ifftshift(delta_H, axes=field.spatial_dims)
    return delta_H


def transform_propagate_sas(
    field: Field,
    z: ScalarLike,
    n: ScalarLike,
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
        z: How far to propagate as a scalar value in units of distance.
        n: A float that defines the (isotropic) refractive index of the medium.
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
    sz = np.array(field.spatial_shape)
    pad_width = tuple(sz // pad_factor)
    field = pad(field, pad_width, cval=cval)

    def _forward(field: Field, z: ScalarLike) -> tuple[Array, Array]:
        delta_H = compute_sas_precompensation(field, z, n)
        field = kernel_propagate(field, delta_H)
        field = transform_propagate(
            field, z, n, 0, 0, skip_initial_phase, skip_final_phase
        )
        return field.u, field.dx

    def _inverse(field: Field, z: ScalarLike) -> tuple[Array, Array]:
        field = transform_propagate(
            field, z, n, 0, 0, skip_initial_phase, skip_final_phase
        )
        delta_H = compute_sas_precompensation(field, z, n)
        field = kernel_propagate(field, delta_H)
        return field.u, field.dx

    u, dx = jax.lax.cond(z >= 0, _forward, _inverse, field, z)
    field = field.replace(u=u, dx=dx)
    return crop(field, pad_width)


def transfer_propagate(
    field: Field,
    z: ScalarLike | Float[Array, "z"],
    n: ScalarLike,
    pad_width: int,
    cval: float = 0,
    absorbing_boundary: Literal["tukey", "super_gaussian"] | None = None,
    absorbing_boundary_width: float = 0.65,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    shift_yx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    output_dx: ArrayLike | None = None,
    output_shape: tuple[int, int] | None = None,
    use_czt: bool = True,
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Fresnel propagate ``field`` for a distance ``z`` using transfer method. This
    method is also called the convolutional Fresnel propagation (CV-FR) method.

    Args:
        field: ``Field`` to be propagated.
        z: How far to propagate as either a scalar value in units of distance
            or a 1D array of distances (in which case a batch dimension will be
            added to the resulting ``Field``).
        n: A float that defines the (isotropic) refractive index of the medium.
        pad_width: A keyword argument integer defining the pad length for
            the propagation FFT. Use padding calculator utilities from
            ``chromatix.functional.propagation`` to compute the padding.
            !!! warning
                The pad value hould not be a Jax array, otherwise a
                ConcretizationError will arise when traced!
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        absorbing_boundary: An optional string that determines which absorbing
            boundary condition is applied (either "tukey" or "super_gaussian",
            for the Tukey or super Gaussian pupils respectively). Either choice
            will taper the propagated field to 0 at the edges to reduce aliasing
            at the edges due to wrapping. Defaults to None in which case no
            absorbing boundary is applied.
        absorbing_boundary: A float determining the diameter (as a percentage)
            of the propagated field that will be permitted without being
            absorbed. The edges of the field beyond this boundary will taper
            smoothly to 0 using the chosen boundary function.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        shift_yx: If provided, defines a shift in the destination
            plane. Should be an array of shape `[2,]` in the format `[y, x]`.
        output_dx: If provided, defines a different output sampling at the output
            plane.
        output_shape: If provided, defines the output shape of the field. Should be
            a tuple of integers. If not provided and ``dx`` is provided, the
            output shape will default to that of the input field.
        use_czt: Whether or not to use chirp Z-transform for different output
            sampling. Defaults to True if `output_dx` or `output_shape` is provided, and
            to False if neither is provided.
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    field = pad(field, pad_width, cval=cval)
    if output_dx is None and output_shape is None:
        # If neither output_dx nor output_shape is provided, use the default ASM propagation
        # as FFT is faster than CZT
        use_czt = False
    z = jnp.atleast_1d(z)
    # assert field.num_batch_dims == 0 or field.batch_dims[-1] == z.size, (
    #     "Must have no batch dimensions or innermost batch dimension must have size z"
    # )
    propagator = compute_transfer_propagator(field, z, n, kykx)
    field = kernel_propagate(
        field,
        propagator,
        absorbing_boundary=absorbing_boundary,
        absorbing_boundary_width=absorbing_boundary_width,
        output_dx=output_dx,
        output_shape=output_shape,
        shift_yx=shift_yx,
        use_czt=use_czt,
    )
    if mode == "same":
        field = crop(field, pad_width)
    return field


def asm_propagate(
    field: Field,
    z: ScalarLike | Float[Array, "z"],
    n: ScalarLike,
    pad_width: int,
    cval: float = 0,
    absorbing_boundary: Literal["tukey", "super_gaussian"] | None = None,
    absorbing_boundary_width: float = 0.65,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    remove_evanescent: bool = False,
    bandlimit: bool = False,
    shift_yx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    output_dx: ArrayLike | None = None,
    output_shape: tuple[int, int] = None,
    use_czt: bool = True,
    mode: Literal["full", "same"] = "full",
) -> Field:
    """
    Propagate ``field`` for a distance ``z`` using angular spectrum method.

    This method does not remove evanescent waves.

    Args:
        field: ``Field`` to be propagated.
        z: How far to propagate as either a scalar value in units of distance
            or a 1D array of distances (in which case a batch dimension will be
            added to the resulting ``Field``).
        n: A float that defines the (isotropic) refractive index of the medium.
        pad_width: A keyword argument integer defining the pad length for
            the propagation FFT. Use padding calculator utilities from
            ``chromatix.functional.propagation`` to compute the padding.
            !!! warning
                The pad value hould not be a Jax array, otherwise a
                ConcretizationError will arise when traced!
        cval: The background value to use when padding the Field. Defaults to 0
            for zero padding.
        absorbing_boundary: An optional string that determines which absorbing
            boundary condition is applied (either "tukey" or "super_gaussian",
            for the Tukey or super Gaussian pupils respectively). Either choice
            will taper the propagated field to 0 at the edges to reduce aliasing
            at the edges due to wrapping. Defaults to None in which case no
            absorbing boundary is applied.
        absorbing_boundary: A float determining the diameter (as a percentage)
            of the propagated field that will be permitted without being
            absorbed. The edges of the field beyond this boundary will taper
            smoothly to 0 using the chosen boundary function.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format `[ky, kx]`.
        remove_evanescent: If ``True``, removes evanescent waves. Defaults to
            False.
        bandlimit: If ``True``, bandlimited the kernel according to "Band-
            Limited Angular Spectrum Method for Numerical Simulation of Free-
            Space Propagation in Far and Near Fields" (2009) by Matsushima and
            Shimobaba. Defaults to ``False``.
        shift_yx: If provided, defines a shift in the destination
            plane. Should be an array of shape `[2,]` in the format `[y, x]`.
        output_dx: If provided, defines a different output sampling at the output
            plane.
        output_shape: If provided, defines the output shape of the field. Should be
            a tuple of integers. If not provided and ``dx`` is provided, the
            output shape will default to that of the input field.
        use_czt: Whether or not to use chirp Z-transform for different output
            sampling. Defaults to True if `output_dx` or `output_shape` is provided, and
            to False if neither is provided.
        mode: Either "full" or "same". If "same", the shape of the output
            ``Field`` will match the shape of the incoming ``Field``. Defaults
            to "full", in which case the output shape will include padding.
    """
    field = pad(field, pad_width, cval=cval)
    if output_dx is None and output_shape is None:
        # If neither output_dx nor output_shape is provided, use the default ASM propagation
        # as FFT is faster than CZT
        use_czt = False
    propagator = compute_asm_propagator(
        field,
        z,
        n,
        kykx,
        bandlimit,
        shift_yx if not use_czt else (0.0, 0.0),
        remove_evanescent=remove_evanescent,
    )
    field = kernel_propagate(
        field,
        propagator,
        absorbing_boundary=absorbing_boundary,
        absorbing_boundary_width=absorbing_boundary_width,
        output_dx=output_dx,
        output_shape=output_shape,
        shift_yx=shift_yx,
        use_czt=use_czt,
    )
    if mode == "same":
        field = crop(field, pad_width)
    return field


def kernel_propagate(
    field: Field,
    propagator: ArrayLike,
    absorbing_boundary: Literal["tukey", "super_gaussian"] | None = None,
    absorbing_boundary_width: float = 0.65,
    output_dx: ArrayLike | None = None,
    output_shape: tuple[int, int] | None = None,
    shift_yx: tuple[float, float] | Float[Array, "2"] = (0.0, 0.0),
    use_czt: bool = False,
) -> Field:
    """
    Propagate an incoming ``Field`` by the given propagation kernel
    (``propagator``). This amounts to performing a Fourier convolution of the
    ``field`` and the ``propagator``. Can optionally apply an absorbing boundary
    (a tapered pupil function) to the field after propagation.

    Args:
        field: ``Field`` to be propagated.
        propagator: The propagation kernel.
        absorbing_boundary: An optional string that determines which absorbing
            boundary condition is applied (either "tukey" or "super_gaussian",
            for the Tukey or super Gaussian pupils respectively). Either choice
            will taper the propagated field to 0 at the edges to reduce aliasing
            at the edges due to wrapping. Defaults to None in which case no
            absorbing boundary is applied.
        absorbing_boundary: A float determining the diameter (as a percentage)
            of the propagated field that will be permitted without being
            absorbed. The edges of the field beyond this boundary will taper
            smoothly to 0 using the chosen boundary function.
    """
    _boundaries = {"tukey": tukey_pupil, "super_gaussian": super_gaussian_pupil}
    assert absorbing_boundary is None or absorbing_boundary in _boundaries, (
        f"The absorbing_boundary must be None or in {_boundaries.keys()}."
    )
    axes = field.spatial_dims
    shift_yx = jnp.asarray(shift_yx)
    if output_dx is None and output_shape is None and not use_czt:
        # shifting accounted for in `propagator`
        u = jnp.fft.ifft2(jnp.fft.fft2(field.u, axes=axes) * propagator, axes=axes)
        field = shift_grid(field, shift_yx)
    else:
        if output_shape is None:
            output_shape = field.spatial_shape
        if output_dx is None:
            output_dx = field.dx
        in_field = field.u
        in_field_df = field.df
        in_field_f_grid = field.f_grid
        in_field_extent = field.extent.squeeze()
        field = shift_grid(field, shift_yx)
        field = Field.empty_like(field, dx=output_dx, shape=output_shape)

        # Scaling factor in Eq 7 of "Band-limited angular spectrum numerical
        # propagation method with selective scaling of observation window size
        # and sample number"
        alpha = field.dx / in_field_df

        # output field in k-space
        # u = fft(in_field, axes=axes, shift=True) * jnp.fft.fftshift(propagator, axes=axes)
        u = jnp.fft.fftshift(
            propagator
            * jnp.fft.fft2(jnp.fft.ifftshift(in_field, axes=axes), axes=axes),
            axes=axes,
        )

        if use_czt:
            (y_min, y_max), (x_min, x_max) = field.spatial_limits
            limits_min = [y_min, x_min]
            limits_max = [y_max, x_max]
            T = in_field_extent

            # loop over dimensions
            for d in range(len(axes)):
                # -- chirp z-transform
                m = output_shape[d]
                a = jnp.exp(-1j * 2 * jnp.pi / T[d] * limits_min[d])
                w = jnp.exp(
                    1j * (2 * jnp.pi / T[d]) * (limits_max[d] - limits_min[d]) / (m - 1)
                )
                u = czt(x=u, m=m, a=a, w=w, axis=axes[d])

                # -- modulate
                N = (m - 1) // 2
                u = jnp.moveaxis(u, axes[d], -1)
                C = w ** (-N * jnp.arange(m)) * (a**N)
                u *= C  # applied to last dimension
                u = jnp.moveaxis(u, -1, axes[d])

            u *= jnp.prod(1 / alpha)

        else:
            # Eq 9 of "Band-limited angular spectrum numerical propagation method
            # with selective scaling of observation window size and sample number"
            # (2012)
            wn = alpha * in_field_f_grid
            f = jnp.prod(jnp.exp(-1j * jnp.pi / alpha * wn**2), axis=-1)
            B = u * jnp.prod(
                (1 / alpha) * jnp.exp(1j * jnp.pi / alpha * wn**2), axis=-1
            )
            mod_terms = jnp.prod(
                field.dx * jnp.exp(1j * jnp.pi / alpha * field.grid**2),
                axis=-1,
            )
            u = mod_terms * fftconvolve(B, f, mode="same", axes=axes)

    if absorbing_boundary is not None:
        pupil = _boundaries[absorbing_boundary]
        field = pupil(field, absorbing_boundary_width)
    return field.replace(u=u)


def compute_transfer_propagator(
    field: Field,
    z: ScalarLike | Float[Array, "z"],
    n: ScalarLike,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
) -> Array:
    """
    Compute propagation kernel for Fresnel propagation.
    Returns an array that can be multiplied with the Fourier transform of the
    incoming Field, as performed by kernel_propagate.

    Args:
        field: ``Field`` to be propagated.
        z: How far to propagate as either a scalar value in units of distance
            or a 1D array of distances (in which case a batch dimension will be
            added to the resulting ``Field``).
        n: A float that defines the (isotropic) refractive index of the medium.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format `[ky, kx]`.
    """
    z = jnp.asarray(z)
    kykx = jnp.asarray(kykx)
    if z.size > 1:
        z = _broadcast_1d_to_innermost_batch(z, field.spatial_dims)
    phase = (
        -jnp.pi * field.broadcasted_wavelength / n * z * l2_sq_norm(field.f_grid - kykx)
    )
    return jnp.fft.ifftshift(jnp.exp(1j * phase), axes=field.spatial_dims)


def compute_asm_propagator(
    field: Field,
    z: ScalarLike | Float[Array, "z"],
    n: ScalarLike,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    bandlimit: bool = False,
    shift_yx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    remove_evanescent: bool = False,
) -> Array:
    """
    Compute propagation kernel for propagation with no Fresnel approximation.

    This version of the propagation kernel does not remove evanescent waves,
    as per the definition of the angular spectrum method. Returns an array
    that can be multiplied with the Fourier transform of the incoming Field, as
    performed by kernel_propagate.

    Args:
        field: ``Field`` to be propagated.
        z: How far to propagate as either a scalar value in units of distance
            or a 1D array of distances (in which case a batch dimension will be
            added to the resulting ``Field``).
        n: A float that defines the (isotropic) refractive index of the medium.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format `[ky, kx]`.
        bandlimit: If ``True``, bandlimited the kernel according to "Band-
            Limited Angular Spectrum Method for Numerical Simulation of Free-
            Space Propagation in Far and Near Fields" (2009) by Matsushima and
            Shimobaba. Defaults to ``False``.
        shift_yx: If provided, defines a shift in the destination
            plane. Should be an array of shape `[2,]` in the format `[y, x]`.
        remove_evanescent: If ``True``, removes evanescent waves. Defaults to
            False.
    """
    # kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    z = jnp.asarray(z)
    kykx = jnp.asarray(kykx)
    shift_yx = jnp.asarray(shift_yx)
    if z.size > 1:
        z = _broadcast_1d_to_innermost_batch(z, field.spatial_dims)
    kernel = 1 - (field.broadcasted_wavelength / n) ** 2 * l2_sq_norm(
        field.f_grid - kykx
    )
    if remove_evanescent:
        delay = jnp.sqrt(jnp.maximum(kernel, 0.0))
    else:
        delay = jnp.sqrt(jnp.complex64(kernel))
    # shift in output plane
    # shift_yx = _broadcast_1d_to_grid(shift_yx, field.ndim)
    out_shift = 2 * jnp.pi * jnp.sum(field.f_grid * shift_yx, axis=-1)
    # compute field
    phase = (
        2 * jnp.pi * (jnp.abs(z) * n / field.broadcasted_wavelength) * delay + out_shift
    )
    kernel_field = jnp.where(z >= 0, jnp.exp(1j * phase), jnp.conj(jnp.exp(1j * phase)))
    if bandlimit:
        # Table 1 of "Shifted angular spectrum method for off-axis numerical
        # propagation" (2010) by Matsushima in vectorized form
        k_limit_p = ((shift_yx + 1 / (2 * field.df)) ** (-2) * z**2 + 1) ** (
            -1 / 2
        ) / field.broadcasted_wavelength
        k_limit_n = ((shift_yx - 1 / (2 * field.df)) ** (-2) * z**2 + 1) ** (
            -1 / 2
        ) / field.broadcasted_wavelength
        k0 = (1 / 2) * (
            jnp.sign(shift_yx + field.extent) * k_limit_p
            + jnp.sign(shift_yx - field.extent) * k_limit_n
        )
        k_width = (
            jnp.sign(shift_yx + field.extent) * k_limit_p
            - jnp.sign(shift_yx - field.extent) * k_limit_n
        )
        k_max = k_width / 2
        # obtain rect filter to bandlimit (Eq. 23)
        H_filter_yx = jnp.abs(field.f_grid - k0) <= k_max
        H_filter = H_filter_yx[..., 0] * H_filter_yx[..., 1]
        # apply filter
        kernel_field = kernel_field * H_filter
    return jnp.fft.ifftshift(kernel_field, axes=field.spatial_dims)


def compute_padding_transform(
    height: int, wavelength: float, dx: float, z: float
) -> int:
    """
    Automatically estimate the padding required for transform propagation.

    Args:
        height: Height (number of pixels in the y direction) of the field. This
            assumes that the field is a square (height is the same as width).
        wavelength: The wavelength of the field as a scalar in units of distance
            (assumed to be a monochromatic field).
        dx: The spacing of the samples of the field as a scalar in units of
            distance. Assumes square pixels.
        z: A float that defines how far to propagate in units of distance.
    """
    # TODO: works only for square fields
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (wavelength * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2
    N = (np.ceil((Q * M) / 2) * 2).astype(int)
    pad_width = (N - M).astype(int)
    return pad_width


def compute_padding_transfer(
    height: int, wavelength: float, dx: float, z: float
) -> int:
    """
    Automatically estimate the padding required for transfer propagation.

    Args:
        height: Height (number of pixels in the y direction) of the field. This
            assumes that the field is a square (height is the same as width).
        wavelength: The wavelength of the field as a scalar in units of distance
            (assumed to be a monochromatic field).
        dx: The spacing of the samples of the field as a scalar in units of
            distance. Assumes square pixels.
        z: A float that defines how far to propagate in units of distance.
    """
    # TODO: works only for square fields
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (wavelength * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2
    N = (np.ceil((Q * M) / 2) * 2).astype(int)
    pad_width = (N - M).astype(int)
    return pad_width


def compute_padding_exact(height: int, wavelength: float, dx: float, z: float) -> int:
    """
    Automatically estimate the padding required for exact/angular wavelength propagation.

    Args:
        height: Height (number of pixels in the y direction) of the field. This
            assumes that the field is a square (height is the same as width).
        wavelength: The wavelength of the field as a scalar in units of distance
            (assumed to be a monochromatic field).
        dx: The spacing of the samples of the field as a scalar in units of
            distance. Assumes square pixels.
        z: A float that defines how far to propagate in units of distance.
    """
    # TODO: works only for square fields
    D = height * dx  # height of field in real coordinates
    Nf = np.max((D / 2) ** 2 / (wavelength * z))  # Fresnel number
    M = height  # height of field in pixels
    Q = 2 * np.maximum(1.0, M / (4 * Nf))  # minimum pad ratio * 2
    scale = np.max((wavelength / (2 * dx)))
    # assert scale < 1, "Can't do exact transfer when field.dx < lambda / 2"
    Q = Q / np.sqrt(1 - scale**2)  # minimum pad ratio for exact transfer
    N = (np.ceil((Q * M) / 2) * 2).astype(int)
    pad_width = (N - M).astype(int)
    return pad_width
