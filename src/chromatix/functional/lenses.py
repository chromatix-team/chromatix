import jax.numpy as jnp

from chromatix import Field
from chromatix.functional.amplitude_masks import amplitude_change
from chromatix.functional.convenience import (
    optical_fft,
    optical_debye_wolf,
    optical_debye_wolf_factored_chunked,
)
from chromatix.functional.phase_masks import phase_change
from chromatix.typing import Array, ScalarLike

from ..utils import l2_sq_norm
from ..utils.initializers import (
    hexagonal_microlens_array_amplitude_and_phase,
    microlens_array_amplitude_and_phase,
    rectangular_microlens_array_amplitude_and_phase,
)
from .pupils import circular_pupil

__all__ = [
    "thin_lens",
    "ff_lens",
    "ff_lens_debye",
    "ff_lens_debye_chunked",
    "df_lens",
    "microlens_array",
    "hexagonal_microlens_array",
    "rectangular_microlens_array",
]


def thin_lens(
    field: Field, f: ScalarLike, n: ScalarLike, NA: ScalarLike | None = None
) -> Field:
    """
    Applies a thin lens placed directly after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` directly after the lens.
    """
    L = jnp.sqrt(field.spectrum * f / n)
    phase = -jnp.pi * l2_sq_norm(field.grid) / L**2

    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    return field * jnp.exp(1j * phase)


def ff_lens(
    field: Field,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike | None = None,
    inverse: bool = False,
) -> Field:
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    # Pupil
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)
    if inverse:
        # if inverse, propagate over negative distance
        f = -f
    return optical_fft(field, f, n)


def ff_lens_debye(
    field: Field,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike | None = None,
    inverse: bool = False,
    range_um: float = 500,
    num_samples: int = 512,
    transverse_bool: bool = False,
) -> Field:
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    # Pupil
    if NA is not None:
        pass
        # print("Applying pupil mask")
        # D = 2 * f * jnp.tan(jnp.arcsin(NA / n))  # Expression for NA yields width of pupil
        # field = circular_pupil(field, D)
    if inverse:
        # if inverse, propagate over negative distance
        f = -f
    oversample_factor = 0.5
    field_size_um = oversample_factor * field.spectrum[0].squeeze() * f
    x_range = field_size_um / field.dx[1].squeeze()
    y_range = field_size_um / field.dx[0].squeeze()
    x_range = range_um
    y_range = range_um
    nx_out = int(x_range / field.dx[1].squeeze() / 10)  #field.shape[2] * oversample_factor
    ny_out = int(y_range / field.dx[0].squeeze() / 10)  #field.shape[1] * oversample_factor
    nx_out = num_samples
    ny_out = num_samples
    print(f"x_range: {x_range:.2f} um, y_range: {y_range:.2f} um, nx_out: {nx_out}, ny_out: {ny_out}")
    return optical_debye_wolf(
        field, f, n, NA=NA,
        nx_out=nx_out,
        ny_out=ny_out,
        x_range=x_range,
        y_range=y_range,
        transverse_bool=transverse_bool,
        )

def ff_lens_debye_chunked(
    field: Field,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike | None = None,
    inverse: bool = False,
    range_um: float = 500,
    num_samples: int = 512,
    transverse_bool: bool = True,
    chunk_size: int = 256,
) -> Field:
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    # Pupil
    if NA is not None:
        pass
        # print("Applying pupil mask")
        # D = 2 * f * jnp.tan(jnp.arcsin(NA / n))  # Expression for NA yields width of pupil
        # field = circular_pupil(field, D)
    if inverse:
        # if inverse, propagate over negative distance
        f = -f
    oversample_factor = 0.5
    field_size_um = oversample_factor * field.spectrum[0].squeeze() * f
    x_range = field_size_um / field.dx[1].squeeze()
    y_range = field_size_um / field.dx[0].squeeze()
    x_range = range_um
    y_range = range_um
    nx_out = (x_range / field.dx[1].squeeze() / 10).astype(jnp.int32)  #field.shape[2] * oversample_factor
    ny_out = (y_range / field.dx[0].squeeze() / 10).astype(jnp.int32)  #field.shape[1] * oversample_factor
    nx_out = num_samples
    ny_out = num_samples
    if False:
        print(f"x_range: {x_range:.2f} um, y_range: {y_range:.2f} um, nx_out: {nx_out}, ny_out: {ny_out}")
    return optical_debye_wolf_factored_chunked(
        field, f, n, NA=NA,
        nx_out=nx_out,
        ny_out=ny_out,
        x_range=x_range,
        y_range=y_range,
        transverse_bool=transverse_bool,
        chunk_size=chunk_size,
        )


def df_lens(
    field: Field,
    d: ScalarLike,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike | None = None,
    inverse: bool = False,
) -> Field:
    """
    Applies a thin lens placed a distance ``d`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        d: Distance from the incoming ``Field`` to the lens.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    if NA is not None:
        D = 2 * f * NA / n  # Expression for NA yields width of pupil
        field = circular_pupil(field, D)

    if inverse:
        # if inverse, propagate over negative distance
        f = -d
        d = -f
    field = optical_fft(field, f, n)

    # Phase factor due to distance d from lens
    L = jnp.sqrt(jnp.complex64(field.spectrum * f / n))  # Lengthscale L
    phase = jnp.pi * (1 - d / f) * l2_sq_norm(field.grid) / jnp.abs(L) ** 2
    return field * jnp.exp(1j * phase)


def microlens_array(
    field: Field,
    n: ScalarLike,
    fs: Array,
    centers: Array,
    radii: Array,
    block_between: bool = False,
) -> Field:
    amplitude, phase = microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n,
        fs,
        centers,
        radii,
    )
    field = phase_change(field, phase)
    if block_between:
        field = amplitude_change(field, amplitude)
    return field


def hexagonal_microlens_array(
    field: Field,
    n: ScalarLike,
    f: Array,
    num_lenses_per_side: ScalarLike,
    radius: Array,
    separation: ScalarLike,
    block_between: bool = False,
) -> Field:
    amplitude, phase = hexagonal_microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n,
        f,
        num_lenses_per_side,
        radius,
        separation,
    )
    field = phase_change(field, phase)
    if block_between:
        field = amplitude_change(field, amplitude)
    return field


def rectangular_microlens_array(
    field: Field,
    n: ScalarLike,
    f: Array,
    num_lenses_height: ScalarLike,
    num_lenses_width: ScalarLike,
    radius: Array,
    separation: ScalarLike,
    block_between: bool = False,
) -> Field:
    amplitude, phase = rectangular_microlens_array_amplitude_and_phase(
        field.spatial_shape,
        field._dx[0, 0],
        field.spectrum[..., 0, 0].squeeze(),
        n,
        f,
        num_lenses_height,
        num_lenses_width,
        radius,
        separation,
    )
    field = phase_change(field, phase)
    if block_between:
        field = amplitude_change(field, amplitude)
    return field


