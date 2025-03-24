from typing import Callable, Optional, Tuple, Union

import jax.numpy as jnp
from chex import PRNGKey

from chromatix.functional.convenience import optical_fft
from chromatix.utils.czt import cztn

from ..field import Field
from ..utils import l2_sq_norm
from .pupils import circular_pupil

__all__ = ["thin_lens", "ff_lens", "df_lens", "ff_lens2"]


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


def ff_lens2(
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
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.

    Returns:
        The ``Field`` propagated a distance ``f`` after the lens.
    """
    zoom_factor = (
        2
        * NA  # 1.3
        * camera_shape[0]  # 256
        * camera_pixel_pitch  # 0.125
        / wavelength  # 0.532
        / (field.shape[1] - 1)  # 255
    )

    # Compute w
    end = zoom_factor * jnp.pi
    start = -zoom_factor * jnp.pi
    m = jnp.array(field.spatial_shape)
    w_phase = (end - start) / (m - 1)
    w = jnp.exp(1j * w_phase)

    # Compute a
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
