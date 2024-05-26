from typing import Optional, Union, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, assert_equal_shape, assert_rank
from ..field import VectorField, ScalarField
from chromatix.field import pad, crop
from ..utils import _broadcast_2d_to_spatial, center_pad
from .propagation import exact_propagate, kernel_propagate, compute_exact_propagator
from .polarizers import polarizer


def sqr_dist_to_line(z, y, x, start, n):
    """
    returns an array with each pixel being assigned to the square distance to that line and an array with the distance along the line
    """
    dx = x - start[2]
    dy = y - start[1]
    dz = z - start[0]
    dot_dn = dx * n[2] + dy * n[1] + dz * n[0]
    return (dx - dot_dn * n[2]) ** 2 + (dy - dot_dn * n[1]) ** 2 + (
        dz - dot_dn * n[0]
    ) ** 2, dot_dn


def draw_line(arr, start, stop, thickness=0.3, intensity=1.0):
    """
    Draw a line in a 3D object with a given thickness and intensity.

    Args:
        obj: The object to draw the line in.
        start: The start of the line.
        end: The end of the line.
        thickness: The thickness of the line.
        intensity: The intensity of the line.
    """
    direction = jnp.subtract(stop, start)
    line_length = jnp.sqrt(jnp.sum(jnp.square(direction)))
    n = direction / line_length

    sigma2 = 2 * thickness**2

    z, y, x = jnp.meshgrid(
        jnp.arange(arr.shape[0]),
        jnp.arange(arr.shape[1]),
        jnp.arange(arr.shape[2]),
        indexing="ij",
    )
    d2, t = sqr_dist_to_line(z, y, x, start, n)

    line_weight = (
        (t > 0) * (t < line_length)
        + (t <= 0) * jnp.exp(-(t**2) / sigma2)
        + (t >= line_length) * jnp.exp(-((t - line_length) ** 2) / sigma2)
    )
    return arr + intensity * jnp.exp(-d2 / sigma2) * line_weight


def filaments3D(
    sz,
    intensity=1.0,
    radius=0.8,
    rand_offset=0.05,
    rel_theta=1.0,
    num_filaments=50,
    apply_seed=True,
    thickness=0.3,
):
    """
    filaments3D(sz; radius = 0.8, rand_offset=0.05, rel_theta=1.0, num_filaments=50, apply_seed=true, thickness=0.8)
    Create a 3D representation of filaments.

    # Arguments
    - sz: A 3D shape tuple representing the size of the object.
    - `radius`: A tuple of real numbers (or a single real number) representing the relative radius of the volume in which the filaments will be created.
        Default is 0.8. If a tuple is used, the filamets will be created in a corresponding elliptical volume.
        Note that the radius is only enforced in the version `filaments3D` which creates the array rather than adding.
    - `rand_offset`: A tuple of real numbers representing the random offsets of the filaments in relation to the size. Default is 0.05.
    - `rel_theta`: A real number representing the relative theta range of the filaments. Default is 1.0.
    - `num_filaments`: An integer representing the number of filaments to be created. Default is 50.
    - `apply_seed`: A boolean representing whether to apply a seed to the random number generator. Default is true.
    - `thickness`: A real number representing the thickness of the filaments in pixels. Default is 0.8.

    The result is added to the obj input array
    """

    sz = jnp.array(sz)

    # Save the state of the rng to reset it after the function is done
    rng_state = np.random.get_state()
    if apply_seed:
        np.random.seed(42)

    # Create the object
    obj = jnp.zeros(sz, dtype=np.float32)

    mid = sz // 2

    # Draw random lines equally distributed over the 3D sphere
    for n in range(num_filaments):
        phi = 2 * jnp.pi * np.random.rand()
        # Theta should be scaled such that the distribution over the unit sphere is uniform
        theta = jnp.arccos(rel_theta * (1 - 2 * np.random.rand()))
        pos = (sz * radius / 2) * jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )
        pos_offset = jnp.array(rand_offset * sz * (np.random.rand(3) - 0.5))
        # Draw line
        obj = draw_line(
            obj,
            pos + pos_offset + mid,
            mid + pos_offset - pos,
            thickness=thickness,
            intensity=intensity,
        )

    # Reset the rng to the state before this function was called
    np.random.set_state(rng_state)
    return obj


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
            defined as ``(D H W)`` array, where D is the total number of slices
        dn_stack: sample refractive index change for each slice ``(D H W)`` array.
            Shape should be the same that for ``absorption_stack``.
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
    """
    assert_equal_shape([absorption_stack, dn_stack])
    field = pad(field, N_pad)
    absorption_stack = center_pad(absorption_stack, (0, N_pad, N_pad))
    dn_stack = center_pad(dn_stack, (0, N_pad, N_pad))
    if propagator is None:
        propagator = compute_exact_propagator(field, thickness_per_slice, n, kykx)
    # NOTE(ac+dd): Unrolling this loop is much faster than ``jax.scan``-likes.
    for absorption, dn in zip(absorption_stack, dn_stack):
        absorption = _broadcast_2d_to_spatial(absorption, field.ndim)
        dn = _broadcast_2d_to_spatial(dn, field.ndim)
        field = thin_sample(field, absorption, dn, thickness_per_slice)
        field = kernel_propagate(field, propagator)
    # Propagate field backwards to the middle (or chosen distance) of the stack
    if reverse_propagate_distance is None:
        reverse_propagate_distance = thickness_per_slice * absorption_stack.shape[0] / 2
    field = exact_propagate(
        field, z=-reverse_propagate_distance, n=n, kykx=kykx, N_pad=0
    )
    return crop(field, N_pad)
