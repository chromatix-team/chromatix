from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
from chex import PRNGKey

from chromatix.functional.convenience import optical_fft
from chromatix.utils.czt import cztn

from ..field import Field
from ..utils import l2_sq_norm
from .pupils import circular_pupil

__all__ = [
    "thin_lens",
    "ff_lens",
    "df_lens",
    "high_na_lens",
    "apply_highNA_basis_change",
]


def thin_lens(field: Field, f: float, n: float, NA: Optional[float] = None) -> Field:
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
    f: float,
    n: float,
    NA: Optional[float] = None,
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


def apply_highNA_basis_change(field: Field, n: float, NA: float) -> Field:
    factor = NA / n
    mask = field.grid[0] ** 2 + field.grid[1] ** 2 <= 1
    sin_theta2 = factor**2 * jnp.sum(field.grid**2, axis=0) * mask
    cos_theta = jnp.sqrt(1 - sin_theta2)
    sin_theta = jnp.sqrt(sin_theta2)

    phi = jnp.arctan2(field.grid[0], field.grid[1])
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = cos_phi**2 - sin_phi**2

    field_x = field.u[:, :, :, :, 2][..., None]
    field_y = field.u[:, :, :, :, 1][..., None]

    e_inf_x = ((cos_theta + 1.0) + (cos_theta - 1.0) * cos_2phi) * field_x + (
        cos_theta - 1.0
    ) * sin_2phi * field_y
    e_inf_y = ((cos_theta + 1.0) - (cos_theta - 1.0) * cos_2phi) * field_y + (
        cos_theta - 1.0
    ) * sin_2phi * field_x
    e_inf_z = -2.0 * sin_theta * (cos_phi * field_x + sin_phi * field_y)

    return field.replace(
        u=jnp.stack([e_inf_z, e_inf_y, e_inf_x], axis=-1).squeeze(-2) / 2
    )


def high_na_lens(
    field: Field,
    NA: Union[float, Callable[[PRNGKey], float]],
    camera_shape: Tuple[int, int],
    camera_pixel_pitch: Union[float, Callable[[PRNGKey], float]],
    wavelength: Union[float, Callable[[PRNGKey], float]],
) -> Field:
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.

    Args:
        field: The ``Field`` to which the lens will be applied.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        camera_shape: The shape of the camera (in pixels).
        camera_pixel_pitch: The pixel pitch of the camera (in microns).
        wavelength: The wavelength of the light (in microns).

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    zoom_factor = (
        2
        * NA
        * camera_shape[0]
        * camera_pixel_pitch
        / wavelength
        / (field.shape[1] - 1)
    )

    # Compute w for chirp z transform
    end = zoom_factor * jnp.pi
    start = -zoom_factor * jnp.pi
    m = jnp.array(field.spatial_shape)
    w_phase = (end - start) / (m - 1)
    w = jnp.exp(1j * w_phase)

    # Compute a for chirp z transform
    a_phase = zoom_factor * jnp.pi
    a = jnp.exp(1j * a_phase)

    return field.replace(
        u=cztn(
            x=field.u,
            m=m,
            a=(a, a),
            w=w,
            axes=field.spatial_dims,
        )
    )


def df_lens(
    field: Field,
    d: float,
    f: float,
    n: float,
    NA: Optional[float] = None,
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
