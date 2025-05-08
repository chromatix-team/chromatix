import jax.numpy as jnp
from jax import Array
from chromatix import ScalarField


# NOTE: this version doesn't normalise power; should be upstreamed to chromatix.
def generic_field(
    dx: float | Array,
    spectrum: float | Array,
    spectral_density: Array,
    amplitude: Array,
    phase: Array,
) -> ScalarField:
    """
    Generates field with arbitrary ``phase`` and ``amplitude``.

    Can also be given ``pupil``.

    Args:
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        amplitude: The amplitude of the field with shape `(B... H W C [1 | 3])`.
        phase: The phase of the field with shape `(B... H W C [1 | 3])`.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    u = amplitude * jnp.exp(1j * phase)
    field = ScalarField.create(dx, spectrum, spectral_density, u=u)
    return field
