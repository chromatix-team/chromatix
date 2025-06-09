import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class Spectrum(eqx.Module, strict=True):
    """
    A spectrum defined by wavelengths and their corresponding densities.
    """

    wavelength: Float[Array, "wv"]
    density: Float[Array, "wv"]

    def __init__(self, wavelength: float | Array, density: float | Array | None = None):
        wavelength = jnp.asarray(wavelength)
        self.wavelength = eqx.error_if(
            wavelength,
            wavelength.ndim != 1,
            "Wavelength must be 1D for polychromatic spectrum.",
        )

        if density is None:
            self.density = jnp.full_like(self.wavelength, 1 / self.wavelength.size)
        else:
            density = jnp.asarray(density)
            density = density / density.sum()
            self.density = eqx.error_if(
                density,
                density.shape != self.wavelength.shape,
                f"Density shape {density.shape} must match wavelength shape {self.wavelength.shape}.",
            )

    @property
    def size(self) -> int:
        return self.wavelength.size


class MonoSpectrum(Spectrum, strict=True):
    """A spectrum with a single wavelength (density is a delta function)."""

    wavelength: Float[Array, "1"]
    density: Float[Array, "1"]

    def __init__(self, wavelength: float | Array):
        self.wavelength = jnp.atleast_1d(jnp.asarray(wavelength))
        assert self.wavelength.shape == (1,), (
            f"Monochromatic must be scalar, got {self.wavelength.shape}"
        )
        self.density = jnp.ones((1,))


def spectrum(
    wavelength: float | Array, density: Array | None = None
) -> Spectrum:
    """Factory that chooses Monochromatic or Polychromatic spectrum."""
    wavelength = jnp.atleast_1d(jnp.asarray(wavelength))

    if wavelength.shape == (1,):
        return MonoSpectrum(wavelength=wavelength)
    else:
        return Spectrum(wavelength=wavelength, density=density)
