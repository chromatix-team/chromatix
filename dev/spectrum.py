import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class AbstractSpectrum(eqx.Module, strict=True):
    wavelength: eqx.AbstractVar[Float[Array, "l"]]
    density: eqx.AbstractVar[Float[Array, "l"]]


class MonochromaticSpectrum(AbstractSpectrum, strict=True):
    """A spectrum with a single wavelength (no real density)."""

    wavelength: Float[Array, "1"]
    density: Float[Array, "1"]

    def __init__(self, wavelength: float | Array):
        self.wavelength = jnp.atleast_1d(jnp.asarray(wavelength))
        assert self.wavelength.shape == (1,), (
            f"Monochromatic must be scalar, got {self.wavelength.shape}"
        )
        self.density = jnp.ones((1,))


class PolyChromaticSpectrum(AbstractSpectrum, strict=True):
    """A spectrum with a single wavelength (no real density)."""

    wavelength: Float[Array, "l"]
    density: Float[Array, "l"]

    def __init__(self, wavelength: float | Array, density: float | Array | None = None):
        wavelength = jnp.asarray(wavelength)
        self.wavelength = eqx.error_if(
            wavelength,
            wavelength.ndim != 1,
            "Wavelength must be 1D for polychromatic spectrum.",
        )

        if density is None:
            self.density = jnp.ones_like(self.wavelength)
        else:
            density = jnp.asarray(density)
            self.density = eqx.error_if(
                density,
                density.shape != self.wavelength.shape,
                f"Density shape {density.shape} must match wavelength shape {self.wavelength.shape}.",
            )


def Spectrum(
    wavelength: float | Array, density: Array | None = None
) -> AbstractSpectrum:
    """Factory that chooses Monochromatic or Polychromatic spectrum."""
    wavelength = jnp.atleast_1d(jnp.asarray(wavelength))

    if wavelength.shape == (1,):
        return MonochromaticSpectrum(wavelength=wavelength)
    else:
        return PolyChromaticSpectrum(wavelength=wavelength, density=density)
