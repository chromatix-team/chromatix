from warnings import deprecated

import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, assert_equal_shape, assert_rank
from einops import rearrange
from jaxtyping import Array, ArrayLike, Float, ScalarLike

from chromatix import (
    ChromaticVectorField,
    Field,
    Monochromatic,
    Vector,
    VectorField,
    crop,
    pad,
)
from chromatix.utils import matvec, outer
from chromatix.utils.fft import fft, ifft

from ..utils import _broadcast_2d_to_spatial, center_pad, l2_sq_norm
from .polarizers import polarizer
from .propagation import (
    compute_asm_propagator,
    kernel_propagate,
)


def jones_sample(
    field: VectorField | ChromaticVectorField,
    absorption: Float[Array, "2 2 h w"],
    dn: Float[Array, "2 2 h w"],
    thickness: ScalarLike,
) -> VectorField | ChromaticVectorField:
    """
    Perturbs an incoming vectorial ``Field`` as if it went through a thin sample
    object with a given ``absorption``, anisotropic refractive index change from
    the refractive index of the medium ``dn``, and ``thickness``. Ignores the
    incoming field in z direction.

    The sample is supposed to follow the thin sample approximation, so the
    sample perturbation is calculated for each component in the Jones matrix as:
    ``exp(1j * 2 * pi * (dn + 1j * absorption) * thickness / lambda)`` where dn
    and absorption are allowed to vary per component of the Jones matrix, but
    thickness is assumed to be the same for each component of the Jones matrix.

    Returns a ``VectorField`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption: The sample absorption defined as a 4D array of shape ``(2 2
            height width)``.
        dn: Sample refractive index change as a 4D array of shape ``(2 2 height
            width)``.
        thickness: Thickness at each sample location as either a scalar or an
            array with shape or shape broadcastable to ``(height width)``.

    Returns:
        ``Field`` immediately after propagating through sample.
    """
    assert_rank(
        absorption,
        4,
        custom_message="Absorption must be array of shape ``(2 2 height width)``",
    )
    assert_rank(
        dn,
        4,
        custom_message="Refractive index must be array of shape ``(2 2 height width)``",
    )
    shape_spec = "m m h w -> m m h w " + ("1 " * (abs(field.spatial_dims[0]) - 2))
    absorption = rearrange(absorption, shape_spec, m=2)
    dn = rearrange(dn, shape_spec, m=2)
    # Thickness is the same for four elements in Jones Matrix
    sample = jnp.exp(
        1j
        * 2
        * jnp.pi
        * (dn + 1j * absorption)
        * thickness
        / field.broadcasted_wavelength
    )
    return polarizer(field, sample[0, 0], sample[0, 1], sample[1, 0], sample[1, 1])


def thin_sample(
    field: Field,
    absorption: Float[Array, "h w"],
    dn: Float[Array, "h w"],
    thickness: ScalarLike | Float[Array, "h w"],
) -> Field:
    """
    Perturbs an incoming ``ScalarField`` as if it went through a thin sample
    object with a given ``absorption``, isotropic refractive index change from
    the refractive index of the medium ``dn``, and ``thickness``.

    The sample is supposed to follow the thin sample approximation, so the
    sample perturbation is calculated as:
    ``exp(1j * 2 * pi * (dn + 1j * absorption) * thickness / lambda)``.

    Returns a ``ScalarField`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption: The sample's absorption defined as a 2D array of shape
            ``(height width)``.
        dn: The sample's isotropic refractive index change as a 2D array of
            shape ``(height width)``.
        thickness: Thickness in units of distance as a scalar or array
            broadcastable to ``(height width)`` (thickness at each sample
            location).
    Returns:
        ``Field`` immediately after propagating through sample.
    """
    assert_rank(
        absorption,
        2,
        custom_message=f"Absorption must have shape {field.spatial_shape}, got {absorption.shape}",
    )
    assert_rank(
        dn,
        2,
        custom_message=f"Refractive index must have shape {field.spatial_shape}, got {dn.shape}.",
    )
    absorption = _broadcast_2d_to_spatial(absorption, field.spatial_dims)
    dn = _broadcast_2d_to_spatial(dn, field.spatial_dims)
    sample = jnp.exp(
        1j
        * 2
        * jnp.pi
        * (dn + 1j * absorption)
        * thickness
        / field.broadcasted_wavelength
    )
    return field * sample


def multislice_thick_sample(
    field: Field,
    absorption_stack: Float[Array, "d h w"],
    dn_stack: Float[Array, "d h w"],
    n: ScalarLike,
    thickness_per_slice: ScalarLike,
    pad_width: int,
    NA: ScalarLike | None = None,
    propagator: ArrayLike | None = None,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    reverse_propagate_distance: ScalarLike | None = None,
    return_stack: bool = False,
    remove_evanescent: bool = False,
    bandlimit: bool = False,
) -> Field:
    """
    Perturbs incoming ``ScalarField`` as if it went through a thick sample. The
    thick sample is modeled as being made of many thin slices each of a given
    thickness. The ``absorption_stack`` and ``dn_stack`` contain the absorbance
    and change in isotropic refractive index from the refractive index of the
    medium of each sample slice. Expects that the same sample is being applied
    to all elements across the batch of the incoming ``Field``.

    A ``propagator`` defining the propagation kernel for the field through each
    slice can be provided. By default, a ``propagator`` is calculated inside
    the function. After passing through all slices, the field is propagated
    backwards to the center of the stack, or by the distances specified by
    ``reverse_propagate_distance`` if provided.

    Returns a ``Field`` with the result of the perturbation.

    !!! warning
        The underlying propagation method now defaults to the angular spectrum
        method (ASM) with ``bandlimit=False`` and ``remove_evanescent=False``.

    Args:
        field: The complex field to be perturbed.
        absorption_stack: The sample's absorption per voxel for each slice
            defined as a 3D array of shape ``(depth height width)``, where
            ``depth`` is the total number of slices.
        dn_stack: The sample's isotropic refractive index change for each slice
            as a 3D array of shape ``(depth height width)``. Shape must be the
            same that for ``absorption_stack``.
        n: Average refractive index of the sample.
        thickness_per_slice: How far to propagate for each slice.
        pad_width: An integer defining the pad length for the
            propagation FFT (NOTE: should not be a `jax` ``Array``, otherwise
            a ConcretizationError will arise when traced!). You can use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            estimate the padding.
        NA: If provided, will be used to define the numerical aperture (limiting
            the captured frequencies) of the lens that is imaging the center of
            the volume. If not provided (the default case), this function will
            return the scattered field directly which may have undesirable high
            frequencies.
        propagator: If provided, will be used as the propagation kernel at
            each plane of the sample. By default, the propagation kernel is
            constructed automatically prior to looping through the planes of
            the sample.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `(2,)` in the format ``ky kx``.
        reverse_propagate_distance: If provided, propagates field at the end
            backwards by this amount from the top of the stack. By default,
            field is propagated backwards to the middle of the sample.
        return_stack: If ``True``, returns the 3D stack of intermediate
            scattered fields at each plane of the thick sample. This 3D
            stack is returned as a ``Field`` where the innermost batch
            dimension is the number of planes in the provided ``dn_stack``/
            ``absorption_stack`` instead of the field defocused to the middle of
            the sample after scattering through the whole sample. If ``True``,
            ``reverse_propagate_distance`` is ignored. Defaults to ``False``.
        remove_evanescent: If ``True``, removes evanescent waves. Defaults to
            ``False``.
        bandlimit: Whether to bandlimit the field before propagation. Defaults
            to ``False``.

    Returns:
        If ``return_stack`` is ``False`` (the default), ``Field`` after
        propagating through sample and being focused to its center (or to any
        plane of the sample defined by ``reverse_propagate_distance``) by an
        optical system with the specified numerical aperture. Otherwise, returns
        the intermediate scattered fields at each plane of the sample (where
        each plane is contained in the first batch dimension of the resulting
        ``Field``).
    """
    assert_equal_shape([absorption_stack, dn_stack])
    field = pad(field, pad_width)
    absorption_stack = center_pad(absorption_stack, (0, pad_width, pad_width))
    dn_stack = center_pad(dn_stack, (0, pad_width, pad_width))
    if propagator is None:
        propagator = compute_asm_propagator(
            field,
            thickness_per_slice,
            n,
            kykx,
            remove_evanescent=remove_evanescent,
            bandlimit=bandlimit,
        )

    def _scatter_through_plane(i: int, u: Array) -> Array:
        absorption = absorption_stack[i]
        dn = dn_stack[i]
        field_i = field.replace(u=u)
        field_i = kernel_propagate(
            field_i,
            propagator,
        )
        field_i = thin_sample(field_i, absorption, dn, thickness_per_slice)
        return field_i.u  # pyright: ignore

    def _accumulate_field_at_each_plane(i: int, fields: Array) -> Array:
        fields = fields.at[i].set(
            _scatter_through_plane(i - 1, field.replace(u=fields[i - 1])).u  # pyright: ignore
        )
        return fields

    if return_stack:
        fields = jnp.zeros((dn_stack.shape[0] + 1,) + field.shape, dtype=field.u.dtype)
        fields = fields.at[0].set(field.u)
        fields = jax.lax.fori_loop(
            1, dn_stack.shape[0] + 1, _accumulate_field_at_each_plane, fields
        )
        field = field.replace(u=jnp.concatenate(fields[1:], axis=-5))
        return crop(field, pad_width)
    else:
        field = field.replace(
            u=jax.lax.fori_loop(0, dn_stack.shape[0], _scatter_through_plane, field.u)
        )
        # Propagate field backwards to the middle (or chosen distance) of the stack
        if reverse_propagate_distance is None:
            reverse_propagate_distance = thickness_per_slice * dn_stack.shape[0] / 2
        defocus_propagator = compute_asm_propagator(
            field, -reverse_propagate_distance, n, kykx=kykx
        )
        if NA is not None:
            # NOTE(dd/2024-12-12): @copypaste(ff_lens) Maybe eventually we should
            # just have some functions for creating masks but not applying them
            # to fields. Here we're creating a custom mask for the NA of the lens
            # imaging the desired plane of the scattering sample.
            mask = l2_sq_norm(field.f_grid) <= (
                (NA / field.broadcasted_wavelength) ** 2
            )
            mask = jnp.fft.ifftshift(mask, axes=field.spatial_dims)
            defocus_propagator *= mask
        field = kernel_propagate(field, defocus_propagator)
        return crop(field, pad_width)


def fluorescent_multislice_thick_sample(
    field: Field,
    fluorescence_stack: Float[Array, "d h w"],
    dn_stack: Float[Array, "d h w"],
    n: ScalarLike,
    thickness_per_slice: ScalarLike,
    pad_width: int,
    key: PRNGKey,
    num_samples: int = 1,
    propagator_forward: ArrayLike | None = None,
    propagator_backward: ArrayLike | None = None,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    remove_evanescent: bool = False,
    bandlimit: bool = False,
) -> Array:
    """
    Perturbs incoming ``Field`` as if it went through a thick, transparent,
    and fluorescent sample, i.e. a sample consisting of some distribution of
    fluorophores emitting within a clear (phase only) scattering volume (with
    isotropic refractive index). The thick sample is modeled as being made
    of many thin slices each of a given thickness. The ``fluorescence_stack``
    contains the fluorescence intensities of each sample slice. The ``dn_stack``
    contains the phase delay as change in refractive index from the refractive
    index of the medium of each sample slice.

    This function simulates the incoherent light from fluorophores using a
    Monte-Carlo approach in which random phases are applied to the fluorescence
    and the resulting propagations are averaged.

    A ``propagator_forward`` and a ``propagator_backward`` defining the
    propagation kernels for the field going forward and backward through each
    slice can be provided. By default, these propagator kernels is calculated
    inside the function. After passing through all slices, the field is
    propagated backwards slice by slice to compute the scattered fluorescence
    intensity.

    Returns an ``Array`` with the result of the scattered fluorescence intensity
    volume.

    !!! warning
        The underlying propagation method now defaults to the angular spectrum
        method (ASM) with ``bandlimit=False`` and ``remove_evanescent=False`.

    Args:
        field: The complex field to be perturbed.
        fluorescence_stack: The sample fluorescence amplitude for each slice
            defined as a 3D array of shape ``(depth height width)``.
        dn_stack: Sample refractive index change as a 3D array of shape``(depth
            height width)``. Shape must be the same as that for
            ``fluorescence_stack``.
        n: Average refractive index of the sample.
        thickness_per_slice: How far to propagate for each slice in units of
            distance.
        pad_width: An integer defining the pad length for the
            propagation FFT (NOTE: should not be a `jax` ``Array``, otherwise
            a ConcretizationError will arise when traced!). You can use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            estimate the padding.
        key: A ``PRNGKey`` used to generate the random phases in each sample.
        num_samples: The number of Monte-Carlo samples (random phases) to simulate.
        propagator_forward: The propagator kernel for the forward propagation through
            the sample.
        propagator_backward: The propagator kernel for the backward propagation
            through the sample.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `(2,)` in the format ``ky kx``.
        remove_evanescent: If ``True``, removes evanescent waves. Defaults to
            ``False``.
        bandlimit: Whether to bandlimit the field before propagation. Defaults
            to ``False``.

    Returns:
        The scattered fluorescence stack as a 3D array of shape ``(depth height
        width)`` after scattering through the provided sample.
    """
    keys = jax.random.split(key, num=num_samples)
    original_field_shape = field.spatial_shape
    axes = field.spatial_dims
    assert_equal_shape([fluorescence_stack, dn_stack])
    field = pad(field, pad_width)
    dn_stack = center_pad(dn_stack, (0, pad_width, pad_width))
    if propagator_forward is None:
        propagator_forward = compute_asm_propagator(
            field,
            thickness_per_slice,
            n,
            kykx,
            remove_evanescent=remove_evanescent,
            bandlimit=bandlimit,
        )
    if propagator_backward is None:
        propagator_backward = compute_asm_propagator(
            field,
            -thickness_per_slice,
            n,
            kykx,
            remove_evanescent=remove_evanescent,
            bandlimit=bandlimit,
        )

    def _forward(
        i: int, field_and_fluorescence_stack: tuple[Field, Float[Array, "d h w"]]
    ) -> tuple[Field, Float[Array, "d h w"]]:
        (field, fluorescence_stack) = field_and_fluorescence_stack
        fluorescence = _broadcast_2d_to_spatial(
            fluorescence_stack[i], field.spatial_dims
        )
        dn = _broadcast_2d_to_spatial(dn_stack[i], field.spatial_dims)
        field = field * jnp.exp(
            1j * 2 * jnp.pi * dn * thickness_per_slice / field.broadcasted_wavelength
        )
        u = ifft(
            fft(fluorescence + field.u, axes=axes, shift=False) * propagator_forward,
            axes=axes,
            shift=False,
        )
        field = field.replace(u=u)
        return field, fluorescence_stack

    def _backward(
        i: int, field_and_intensity_stack: tuple[Field, Float[Array, "d h w"]]
    ) -> tuple[Field, Float[Array, "d h w"]]:
        (field, intensity_stack) = field_and_intensity_stack
        u = field.u * propagator_backward
        field = field.replace(u=u)
        field_i = field.replace(u=ifft(u, axes=axes, shift=False))
        intensity_stack = intensity_stack.at[intensity_stack.shape[0] - 1 - i].add(
            crop(field_i, pad_width).intensity
        )
        return (field, intensity_stack)

    def _sample(
        i: int, field_and_intensity_stack: tuple[Field, Float[Array, "d h w"]]
    ) -> tuple[Field, Float[Array, "d h w"]]:
        (field, intensity_stack) = field_and_intensity_stack
        random_phase_stack = jax.random.uniform(
            keys[i], fluorescence_stack.shape, minval=0, maxval=2 * jnp.pi
        )
        _fluorescence_stack = center_pad(
            fluorescence_stack * jnp.exp(1j * random_phase_stack),
            (0, pad_width, pad_width),
        )
        (field, _) = jax.lax.fori_loop(
            0, _fluorescence_stack.shape[0], _forward, (field, _fluorescence_stack)
        )
        field = field.replace(u=fft(field.u, axes=axes, shift=False))
        (field, intensity_stack) = jax.lax.fori_loop(
            0, intensity_stack.shape[0], _backward, (field, intensity_stack)
        )
        return (field, intensity_stack)

    intensity_stack = jnp.zeros((fluorescence_stack.shape[0], *original_field_shape))
    (_, intensity_stack) = jax.lax.fori_loop(
        0, num_samples, _sample, (field, intensity_stack)
    )
    intensity_stack /= num_samples
    return intensity_stack


def polarized_multislice_thick_sample(
    field: VectorField | ChromaticVectorField,
    potential_stack: Float[Array, "d h w 3 3"],
    n: ScalarLike,
    thickness_per_slice: ScalarLike,
    NA: ScalarLike = 1.0,
) -> VectorField | ChromaticVectorField:
    """Implements a thick sample method for samples with anisotroipc refractive
    index (birefringent samples) per [1].

    [1]: Multislice computational model for birefringent scattering,
    https://doi.org/10.1364/OPTICA.472077.

    Args:
        field: Incoming vectorial field to propagate through sample.
        potential_stack: Scattering potential as a 5D array of shape ``(depth
            height width 3 3)`` where the last two axes of shape `(3, 3)`
            represent the `3 x 3` tensor representing all possible couplings
            of input and output polarization components of the sample at each
            3D location. Note that along each row or column of these matrices
            we expect `(z y x)` order (which matches the `(z y x)` order of our
            vectorial ``Field`` components).
        n: Refractive index of medium.
        thickness_per_slice: Thickness of each slice of the sample along the
            optical axis in units of distance.
        NA: Numerical aperture of the imaging system. Defaults to 1.0.

    Returns:
        ``Field`` just after propagating through sample, imaged through pupil
        with the given numerical aperture.
    """

    def Q_op(u: Array) -> Array:
        # correct
        """Polarization transfer operator"""
        return crop_op(
            ifft(matvec(Q, fft(pad_op(u), axes=spatial_dims)), axes=spatial_dims)
        )

    def H_op(u: Array) -> Array:
        # correct
        """Vectorial scattering operator"""
        prefactor = jnp.where(
            kz > 0,
            -1j / 2 * jnp.exp(1j * kz * thickness_per_slice) / kz * thickness_per_slice,
            0,
        )
        return crop_op(
            ifft(
                matvec(Q, prefactor * fft(pad_op(u), axes=spatial_dims)),
                axes=spatial_dims,
            )
        )

    def P_op(u: Array) -> Array:
        """Vectorial free space operator"""
        prefactor = jnp.where(kz > 0, jnp.exp(1j * kz * thickness_per_slice), 0)
        return crop_op(
            ifft(
                matvec(Q, prefactor * fft(pad_op(u), axes=spatial_dims)),
                axes=spatial_dims,
            )
        )

    def propagate_slice(u: Array, potential_slice: Array) -> tuple[Array, None]:
        scatter_field = matvec(potential_slice, Q_op(u))
        new_field = P_op(u) + H_op(scatter_field)
        return new_field, new_field

    def pad_op(u: Array) -> Array:
        return jnp.pad(u, padding)

    def crop_op(u: Array) -> Array:
        u = u[tuple(cropping)]
        return u

    assert isinstance(field, Vector), "Must be a vectorial Field"
    if not isinstance(field, Monochromatic):
        potential_stack = potential_stack[:, :, :, jnp.newaxis, :, :]
    # Padding and cropping for circular convolution
    padded_shape = 2 * np.array(field.spatial_shape)
    pad_width = padded_shape - np.array(field.spatial_shape)
    spatial_dims = [field.ndim + d for d in field.spatial_dims]
    padding = [(0, 0) for d in range(field.ndim)]
    padding[spatial_dims[0]] = (0, pad_width[0])
    padding[spatial_dims[1]] = (0, pad_width[1])
    cropping = [slice(0, field.shape[d]) for d in range(field.ndim)]
    cropping[spatial_dims[0]] = slice(0, field.spatial_shape[0])
    cropping[spatial_dims[1]] = slice(0, field.spatial_shape[1])

    # Getting k_grid
    k_grid = jnp.fft.ifftshift(pad(field, pad_width // 2).k_grid, axes=spatial_dims)
    km = 2 * jnp.pi * n / field.broadcasted_wavelength
    kz = jnp.sqrt(
        jnp.maximum(0.0, km**2 - l2_sq_norm(k_grid))
    )  # chop off evanescent waves
    k_grid = jnp.concatenate([kz[..., jnp.newaxis], k_grid], axis=-1)

    # Getting PTFT and band limiting
    Q = (-outer(k_grid / km, k_grid / km, in_axis=-1) + jnp.eye(3)).squeeze(-3)
    Q = jnp.where(
        jnp.sum(k_grid[..., 1:, None] ** 2, axis=-2) <= NA**2 * km**2, Q, 0
    )  # Add the NA here

    # Running scan over sample
    u, intermediates = jax.lax.scan(propagate_slice, field.u, potential_stack)
    return field.replace(u=u)


@deprecated(
    "Please switch to the updated function name `polarized_multislice_thick_sample`"
)
def thick_polarized_sample(
    field: VectorField | ChromaticVectorField,
    potential_stack: Float[Array, "d h w 3 3"],
    n: ScalarLike,
    thickness_per_slice: ScalarLike,
    NA: ScalarLike = 1.0,
) -> VectorField | ChromaticVectorField:
    """
    Alias for ``polarized_multislice_thick_sample``. See
    ``polarized_multislice_thick_sample`` for documentation.

    !!! warning
        This alias is deprecated and will be removed in a future release. Please
        switch to the updated name.
    """
    return polarized_multislice_thick_sample(
        field, potential_stack, n, thickness_per_slice, NA=NA
    )
