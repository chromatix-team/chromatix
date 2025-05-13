from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from chex import assert_equal_shape, assert_rank
from jax import Array
from jax.lax import scan
from jax.typing import ArrayLike

from chromatix.field import crop, pad
from chromatix.utils import matvec, outer
from chromatix.utils.fft import fft, ifft

from ..field import ScalarField, VectorField
from ..utils import _broadcast_2d_to_spatial, center_pad
from .polarizers import polarizer
from .propagation import (
    asm_propagate,
    compute_asm_propagator,
    kernel_propagate,
)


def jones_sample(
    field: VectorField, absorption: Array, dn: Array, thickness: Union[float, Array]
) -> VectorField:
    """
    Perturbs an incoming ``VectorField`` as if it went through a thin sample
    object with a given ``absorption``, refractive index change ``dn`` and of
    a given ``thickness`` in the same units as the spectrum of the incoming
    ``VectorField``. Ignores the incoming field in z direction.

    The sample is supposed to follow the thin sample approximation, so the
    sample perturbation is calculated for each component in the Jones matrix as:
    ``exp(1j * 2 * pi * (dn + 1j * absorption) * thickness / lambda)`` where dn
    and absorption are allowed to vary per component of the Jones matrix, but
    thickness is assumed to be the same for each component of the Jones matrix.

    Returns a ``VectorField`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption: The sample absorption defined as ``(2 2 B... H W 1 1)`` array
        dn: Sample refractive index change ``(2 2 B... H W 1 1)`` array
        thickness: Thickness at each sample location as array broadcastable
            to ``(B... H W 1 1)``
    """
    assert_rank(
        absorption,
        field.ndim + 2,
        custom_message="Absorption must be array of shape ``(2 2 B... H W 1 1)``",
    )
    assert_rank(
        dn,
        field.ndim + 2,
        custom_message="Refractive index must be array of shape ``(2 2 B... H W 1 1)``",
    )
    # Thickness is the same for four elements in Jones Matrix
    sample = jnp.exp(
        1j * 2 * jnp.pi * (dn + 1j * absorption) * thickness / field.spectrum
    )
    return polarizer(field, sample[0, 0], sample[0, 1], sample[1, 0], sample[1, 1])


def thin_sample(
    field: ScalarField, absorption: Array, dn: Array, thickness: Union[float, Array]
) -> ScalarField:
    """
    Perturbs an incoming ``ScalarField`` as if it went through a thin sample
    object with a given ``absorption``, refractive index change ``dn`` and of
    a given ``thickness`` in the same units as the spectrum of the incoming
    ``ScalarField``.

    The sample is supposed to follow the thin sample approximation, so the
    sample perturbation is calculated as:
    ``exp(1j * 2 * pi * (dn + 1j * absorption) * thickness / lambda)``.

    Returns a ``ScalarField`` with the result of the perturbation.

    Args:
        field: The complex field to be perturbed.
        absorption: The sample absorption defined as ``(B... H W 1 1)`` array
        dn: Sample refractive index change ``(B... H W 1 1)`` array
        thickness: Thickness at each sample location as array broadcastable
            to ``(B... H W 1 1)``
    """
    assert_rank(
        absorption,
        field.ndim,
        custom_message="Absorption must have same ndim as incoming ``Field``.",
    )
    assert_rank(
        dn,
        field.ndim,
        custom_message="Refractive index must have same ndim as incoming ``Field`.`",
    )
    sample = jnp.exp(
        1j * 2 * jnp.pi * (dn + 1j * absorption) * thickness / field.spectrum
    )
    return field * sample


def multislice_thick_sample(
    field: ScalarField,
    absorption_stack: Array,
    dn_stack: Array,
    n: float,
    thickness_per_slice: float,
    N_pad: int,
    propagator: Optional[Array] = None,
    kykx: Union[Array, Tuple[float, float]] = (0.0, 0.0),
    reverse_propagate_distance: Optional[float] = None,
    remove_evanescent: bool = False,
    bandlimit: bool = False,
) -> ScalarField:
    """
    Perturbs incoming ``ScalarField`` as if it went through a thick sample. The
    thick sample is modeled as being made of many thin slices each of a given
    thickness. The ``absorption_stack`` and ``dn_stack`` contain the absorbance
    and phase delay of each sample slice. Expects that the same sample is being
    applied to all elements across the batch of the incoming ``ScalarField``.

    A ``propagator`` defining the propagation kernel for the field through each
    slice can be provided. By default, a ``propagator`` is calculated inside
    the function. After passing through all slices, the field is propagated
    backwards to the center of the stack, or by the distances specified by
    ``reverse_propagate_distance`` if provided.

    Returns a ``ScalarField`` with the result of the perturbation.

    !!! warning
        The underlying propagation method now defaults to the angular spectrum
        method (ASM) with ``bandlimit=False`` and ``remove_evanescent=False``.

    Args:
        field: The complex field to be perturbed.
        absorption_stack: The sample absorption per micrometre for each slice
            defined as ``(D H W)`` array, where D is the total number of slices.
        dn_stack: sample refractive index change for each slice ``(D H W)`` array.
            Shape should be the same that for ``absorption_stack``.
        n: Average refractive index of the sample.
        thickness_per_slice: How far to propagate for each slice.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a `jax` ``Array``, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            calculate the padding.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `(2,)` in the format ``[ky, kx]``.
        reverse_propagate_distance: If provided, propagates field at the end
            backwards by this amount from the top of the stack. By default,
            field is propagated backwards to the middle of the sample.
        remove_evanescent: If ``True``, removes evanescent waves. Defaults to
            ``False``.
        bandlimit: Whether to bandlimit the field before propagation. Defaults
            to ``False``.
    """
    assert_equal_shape([absorption_stack, dn_stack])
    field = pad(field, N_pad)
    absorption_stack = center_pad(absorption_stack, (0, N_pad, N_pad))
    dn_stack = center_pad(dn_stack, (0, N_pad, N_pad))
    if propagator is None:
        propagator = compute_asm_propagator(
            field,
            thickness_per_slice,
            n,
            kykx,
            remove_evanescent=remove_evanescent,
            bandlimit=bandlimit,
        )
    # NOTE(ac+dd): Unrolling this loop is much faster than ``jax.scan``-likes.
    for absorption, dn in zip(absorption_stack, dn_stack):
        absorption = _broadcast_2d_to_spatial(absorption, field.ndim)
        dn = _broadcast_2d_to_spatial(dn, field.ndim)
        field = thin_sample(field, absorption, dn, thickness_per_slice)
        field = kernel_propagate(field, propagator)
    # Propagate field backwards to the middle (or chosen distance) of the stack
    if reverse_propagate_distance is None:
        reverse_propagate_distance = thickness_per_slice * absorption_stack.shape[0] / 2
    field = asm_propagate(
        field,
        z=-reverse_propagate_distance,
        n=n,
        kykx=kykx,
        N_pad=0,
        remove_evanescent=remove_evanescent,
        bandlimit=bandlimit,
    )
    return crop(field, N_pad)


def fluorescent_multislice_thick_sample(
    field: ScalarField,
    fluorescence_stack: Array,
    dn_stack: Array,
    n: float,
    thickness_per_slice: float,
    N_pad: int,
    key: jax.random.PRNGKey,
    num_samples: int = 1,
    propagator_forward: Optional[Array] = None,
    propagator_backward: Optional[Array] = None,
    remove_evanescent: bool = False,
    bandlimit: bool = False,
    kykx: Union[Array, Tuple[float, float]] = (0.0, 0.0),
) -> Array:
    """
    Perturbs incoming ``ScalarField`` as if it went through a thick,
    transparent, and fluorescent sample, i.e. a sample consisting of some
    distribution of fluorophores emitting within a clear (phase only) scattering
    volume. The thick sample is modeled as being made of many thin slices each
    of a given thickness. The ``fluorescence_stack`` contains the fluorescence
    intensities of each sample slice. The ``dn_stack`` contains the phase delay
    of each sample slice. Expects that the same sample is being applied to all
    elements across the batch of the incoming ``ScalarField``.

    This function simulates the incoherent light from fluorophores using a
    Monte-Carlo approach in which random phases are applied to the fluorescence
    and the resulting propagations are averaged.

    A ``propagator_forward`` and a ``propagator_backward`` defining the
    propagation kernels for the field going forward and backward through each
    slice can be provided. By default, these propagator kernels is calculated
    inside the function. After passing through all slices, the field is
    propagated backwards slice by slice to compute the scattered fluorescence
    intensity.

    Returns an ``Array`` with the result of the scattered fluorescence volume.

    !!! warning
        The underlying propagation method now defaults to the angular spectrum
        method (ASM) with ``bandlimit=False`` and ``remove_evanescent=False`.

    Args:
        field: The complex field to be perturbed.
        fluorescence_stack: The sample fluorescence amplitude for each slice
            defined as ``(D H W)`` array, where D is the total number of slices.
        dn_stack: sample refractive index change for each slice ``(D H W)``
            array. Shape should be the same that for ``fluorescence_stack``.
        n: Average refractive index of the sample.
        thickness_per_slice: How far to propagate for each slice.
        N_pad: A keyword argument integer defining the pad length for the
            propagation FFT (NOTE: should not be a `jax` ``Array``, otherwise
            a ConcretizationError will arise when traced!). Use padding
            calculator utilities from ``chromatix.functional.propagation`` to
            calculate the padding.
        key: A ``PRNGKey`` used to generate the random phases in each sample.
        num_samples: The number of Monte-Carlo samples (random phases) to simulate.
        propagator_forward: The propagator kernel for the forward propagation through
            the sample.
        propagator_backward: The propagator kernel for the backward propagation
            through the sample.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `(2,)` in the format ``[ky, kx]``.
        remove_evanescent: If ``True``, removes evanescent waves. Defaults to
            ``False``.
        bandlimit: Whether to bandlimit the field before propagation. Defaults
            to ``False``.
    """
    keys = jax.random.split(key, num=num_samples)
    original_field_shape = field.shape
    axes = field.spatial_dims
    assert_equal_shape([fluorescence_stack, dn_stack])
    field = pad(field, N_pad)
    dn_stack = center_pad(dn_stack, (0, N_pad, N_pad))
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

    def _forward(i, field_and_fluorescence_stack):
        (field, fluorescence_stack) = field_and_fluorescence_stack
        fluorescence = _broadcast_2d_to_spatial(fluorescence_stack[i], field.ndim)
        dn = _broadcast_2d_to_spatial(dn_stack[i], field.ndim)
        field = field * jnp.exp(
            1j * 2 * jnp.pi * dn * thickness_per_slice / field.spectrum
        )
        u = ifft(
            fft(fluorescence + field.u, axes=axes, shift=False) * propagator_forward,
            axes=axes,
            shift=False,
        )
        field = field.replace(u=u)
        return field, fluorescence_stack

    def _backward(i, field_and_intensity_stack):
        (field, intensity_stack) = field_and_intensity_stack
        u = field.u * propagator_backward
        field = field.replace(u=u)
        field_i = field.replace(u=ifft(u, axes=axes, shift=False))
        intensity_stack = intensity_stack.at[intensity_stack.shape[0] - 1 - i].add(
            crop(field_i, N_pad).intensity[0]
        )
        return (field, intensity_stack)

    def _sample(i, field_and_intensity_stack):
        (field, intensity_stack) = field_and_intensity_stack
        random_phase_stack = jax.random.uniform(
            keys[i], fluorescence_stack.shape, minval=0, maxval=2 * jnp.pi
        )
        _fluorescence_stack = center_pad(
            fluorescence_stack * jnp.exp(1j * random_phase_stack), (0, N_pad, N_pad)
        )
        (field, _) = jax.lax.fori_loop(
            0, _fluorescence_stack.shape[0], _forward, (field, _fluorescence_stack)
        )
        field = field.replace(u=fft(field.u, axes=axes, shift=False))
        (field, intensity_stack) = jax.lax.fori_loop(
            0, intensity_stack.shape[0], _backward, (field, intensity_stack)
        )
        return (field, intensity_stack)

    intensity_stack = jnp.zeros(
        (fluorescence_stack.shape[0], *original_field_shape[1:])
    )
    (_, intensity_stack) = jax.lax.fori_loop(
        0, num_samples, _sample, (field, intensity_stack)
    )
    intensity_stack /= num_samples
    return intensity_stack


def thick_polarised_sample(
    field: VectorField,
    potential: ArrayLike,
    n_background: ArrayLike,
    dz: ArrayLike,
    NA: float = 1.0,
) -> VectorField:
    """Implements a thick sample method polarised samples per
    'Multislice computational model for birefringent scattering'
    https://doi.org/10.1364/OPTICA.472077

    Args:
        field (VectorField): _description_
        potential (ArrayLike): _description_
        n_background (ArrayLike): _description_
        dz (ArrayLike): _description_
        NA (float, optional): _description_. Defaults to 1.0.

    Returns:
        VectorField: _description_
    """

    def Q_op(u: Array) -> Array:
        # correct
        """Polarisation transfer operator"""
        return crop(ifft(matvec(Q, fft(pad(u)))))

    def H_op(u: Array) -> Array:
        # correct
        """Vectorial scattering operator"""
        prefactor = jnp.where(kz > 0, -1j / 2 * jnp.exp(1j * kz * dz) / kz * dz, 0)
        return crop(ifft(matvec(Q, prefactor * fft(pad(u)))))

    def P_op(u: Array) -> Array:
        """Vectorial free space operator"""
        prefactor = jnp.where(kz > 0, jnp.exp(1j * kz * dz), 0)
        return crop(ifft(matvec(Q, prefactor * fft(pad(u)))))

    def propagate_slice(u: Array, potential_slice: Array) -> tuple[Array, None]:
        scatter_field = matvec(potential_slice, Q_op(u))
        new_field = P_op(u) + H_op(scatter_field)
        return new_field, new_field

    def pad(u):
        return jnp.pad(u, padding)

    def crop(u):
        return u[:, : field.spatial_shape[0], : field.spatial_shape[1]]

    # Padding for circular convolution
    padded_shape = 2 * np.array(field.spatial_shape)
    n_pad = padded_shape - np.array(field.spatial_shape)
    padding = ((0, 0), (0, n_pad[0]), (0, n_pad[1]), (0, 0), (0, 0))

    # Getting k_grid
    k_grid = (
        2
        * jnp.pi
        * jnp.stack(
            jnp.meshgrid(
                jnp.fft.fftfreq(n=padded_shape[0], d=field.dx.squeeze()[0]),
                jnp.fft.fftfreq(n=padded_shape[1], d=field.dx.squeeze()[1]),
                indexing="ij",
            )
        )[:, None, ..., None, None]
    )
    km = 2 * jnp.pi * n_background / field.spectrum
    kz = jnp.sqrt(
        jnp.maximum(0.0, km**2 - jnp.sum(k_grid**2, axis=0))
    )  # chop off evanescent waves
    k_grid = jnp.concatenate([kz[None, ...], k_grid], axis=0)

    # Getting PTFT and band limiting
    Q = (-outer(k_grid / km, k_grid / km, in_axis=0) + jnp.eye(3)).squeeze(-3)
    Q = jnp.where(
        jnp.sum(k_grid[1:, ..., None] ** 2, axis=0) <= NA**2 * km**2, Q, 0
    )  # Add the NA here

    # Running scan over sample
    u, intermediates = scan(propagate_slice, field.u, potential)
    return field.replace(u=u)
