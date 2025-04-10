from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
from chex import PRNGKey

from chromatix.field import ScalarField, VectorField, cartesian_to_spherical
from chromatix.functional.convenience import optical_fft
from chromatix.utils.czt import custom_fftn

from ..field import Field
from ..utils import l2_sq_norm
from .pupils import circular_pupil

__all__ = [
    "thin_lens",
    "ff_lens",
    "df_lens",
    "high_na_lens",
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


def high_na_lens(
    field: Field,
    f: float,
    n: float,
    NA: float,
    output_shape: Tuple[int, int],
    output_dx: Union[float, Callable[[PRNGKey], float]],
    wavelength: Union[float, Callable[[PRNGKey], float]],
    return_spherical_u: bool = False,
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
    if field.shape[-1] == 1:
        spherical_u = field.u
    else:
        spherical_u = cartesian_to_spherical(field, n, NA, f)

    zoom_factor = (
        2 * NA * output_shape[0] * output_dx / wavelength / (field.shape[1] - 1)
    )
    u = custom_fftn(
        x=spherical_u,
        k_start=-zoom_factor * jnp.pi,
        k_end=zoom_factor * jnp.pi,
        output_shape=output_shape,
        include_end=True,
        axes=field.spatial_dims,
    )

    create = ScalarField.create if field.shape[-1] == 1 else VectorField.create
    out_field = create(
        output_dx, field.spectrum, field.spectral_density, shape=output_shape
    )
    if return_spherical_u:
        return out_field.replace(u=u), spherical_u
    else:
        return out_field.replace(u=u)


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
