import jax
import jax.numpy as jnp
from chex import PRNGKey, assert_equal_shape, assert_rank
from jax import Array

from chromatix.field import crop, pad
from chromatix.typing import ArrayLike, ScalarLike
from chromatix.utils.fft import fft, ifft

from ..field import ScalarField, VectorField
from ..utils import _broadcast_2d_to_spatial, center_pad
from .polarizers import polarizer
from .propagation import (
    compute_asm_propagator,
    exact_propagate,
    asm_propagate,
    kernel_propagate,
)


def jones_sample(
    field: VectorField, absorption: ArrayLike, dn: ArrayLike, thickness: ScalarLike
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
    field: ScalarField, absorption: ArrayLike, dn: ArrayLike, thickness: ScalarLike
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
    absorption_stack: ArrayLike,
    dn_stack: ArrayLike,
    n: ScalarLike,
    thickness_per_slice: ScalarLike,
    N_pad: int,
    propagator: ArrayLike | None = None,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    reverse_propagate_distance: ScalarLike | None = None,
    return_stack: bool = False,
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
        return_stack: If ``True``, returns the 3D stack of intermediate
            scattered fields at each plane of the thick sample. This 3D
            stack is returned as a ``Field`` where the innermost batch
            dimension is the number of planes in the provided ``dn_stack``/
            ``absorption_stack`` instead of the field defocused to the middle of
            the sample after scattering through the whole sample. If ``True``,
            ``reverse_propagate_distance`` is ignored. Defaults to ``False``.
    """
    assert_equal_shape([absorption_stack, dn_stack])
    field = pad(field, N_pad)
    absorption_stack = center_pad(absorption_stack, (0, N_pad, N_pad))
    dn_stack = center_pad(dn_stack, (0, N_pad, N_pad))
    if propagator is None:
       propagator = compute_asm_propagator(field, thickness_per_slice, n, kykx)
    # NOTE(ac+dd): Unrolling this loop is much faster than ``jax.scan``-likes.
    if return_stack:
        _fields = []
    for absorption, dn in zip(absorption_stack, dn_stack):
        absorption = _broadcast_2d_to_spatial(absorption, field.ndim)
        dn = _broadcast_2d_to_spatial(dn, field.ndim)
        field = thin_sample(field, absorption, dn, thickness_per_slice)
        field = kernel_propagate(field, propagator)
        if return_stack:
            _fields.append(field.u)  # pyright: ignore
    if return_stack:
        field = field.replace(u=jnp.concatenate(_fields, axis=-5))  # pyright: ignore
        return crop(field, N_pad)
    # Propagate field backwards to the middle (or chosen distance) of the stack
    if reverse_propagate_distance is None:
        reverse_propagate_distance = thickness_per_slice * absorption_stack.shape[0] / 2
    field = exact_propagate(
        field, z=-reverse_propagate_distance, n=n, kykx=kykx, N_pad=0
    )
    return crop(field, N_pad)


def fluorescent_multislice_thick_sample(
    field: ScalarField,
    fluorescence_stack: Array,
    dn_stack: Array,
    n: ScalarLike,
    thickness_per_slice: ScalarLike,
    N_pad: int,
    key: PRNGKey,
    num_samples: int = 1,
    propagator_forward: ArrayLike | None = None,
    propagator_backward: ArrayLike | None = None,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
) -> ScalarField:
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
    """
    keys = jax.random.split(key, num=num_samples)
    original_field_shape = field.shape
    axes = field.spatial_dims
    assert_equal_shape([fluorescence_stack, dn_stack])
    field = pad(field, N_pad)
    dn_stack = center_pad(dn_stack, (0, N_pad, N_pad))
    if propagator_forward is None:
        propagator_forward = compute_asm_propagator(field, thickness_per_slice, n, kykx)
    if propagator_backward is None:
        propagator_backward = compute_asm_propagator(
            field, -thickness_per_slice, n, kykx
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


# depolarised wave
def PTFT(k: ArrayLike, km: ScalarLike) -> Array:
    Q = jnp.zeros((3, 3, *k.shape[1:]))

    # Setting diagonal
    Q_diag = 1 - k**2 / km**2
    Q = Q.at[jnp.diag_indices(3)].set(Q_diag)

    # Calculating off-diagonal elements
    def q_ij(i, j):
        return -k[i] * k[j] / km**2

    # Setting upper diagonal
    Q = Q.at[0, 1].set(q_ij(0, 1))
    Q = Q.at[0, 2].set(q_ij(0, 2))
    Q = Q.at[1, 2].set(q_ij(1, 2))

    # Setting lower diagonal, mirror symmetry
    Q = Q.at[1, 0].set(q_ij(0, 1))
    Q = Q.at[2, 0].set(q_ij(0, 2))
    Q = Q.at[2, 1].set(q_ij(1, 2))

    # We move the axes to the back, easier matmul
    return jnp.moveaxis(Q.squeeze(-1), (0, 1), (-2, -1))


def bmatvec(a, b):
    return jnp.matmul(a, b[..., None]).squeeze(-1)


def thick_sample_vector(
    field: VectorField, scatter_potential: ArrayLike, dz: ScalarLike, n: ScalarLike
) -> VectorField:
    def P_op(u: Array) -> Array:
        phase_factor = jnp.exp(1j * kz * dz)
        return ifft(bmatvec(Q, phase_factor * fft(u)))

    def Q_op(u: Array) -> Array:
        return ifft(bmatvec(Q, fft(u)))

    def H_op(u: Array) -> Array:
        phase_factor = -1j * dz / 2 * jnp.exp(1j * kz * dz) / kz
        return ifft(bmatvec(Q, phase_factor * fft(u)))

    # Calculating k vector and PTFT
    # We shift k to align in k-space so we dont need shift just like Q
    km = 2 * jnp.pi * n / field.spectrum
    k = jnp.fft.ifftshift(field.k_grid, axes=field.spatial_dims)
    kz = jnp.sqrt(km**2 - jnp.sum(k**2, axis=0))
    k = jnp.concatenate([kz[None, ...], k], axis=0)
    Q = PTFT(k, km)

    def propagate_slice(u, potential_slice):
        scatter_field = bmatvec(potential_slice, Q_op(u))
        return P_op(u) + H_op(scatter_field), None

    u, _ = jax.lax.scan(propagate_slice, field.u, scatter_potential)
    return field.replace(u=u)
