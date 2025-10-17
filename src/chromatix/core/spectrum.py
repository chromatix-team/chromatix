from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, ScalarLike

from chromatix.typing import wv


class Spectrum(eqx.Module):
    """
    A spectrum defined by wavelengths and their corresponding densities.
    """

    wavelength: Float[Array, "wv"]
    density: Float[Array, "wv"]

    def __init__(self, wavelength: float | Array, density: float | Array | None = None):
        wavelength = jnp.atleast_1d(jnp.asarray(wavelength))
        self.wavelength = eqx.error_if(
            wavelength,
            wavelength.ndim != 1,
            "Wavelength must be 1D for polychromatic spectrum.",
        )

        if density is None:
            self.density = jnp.full_like(self.wavelength, 1 / self.wavelength.size)
        else:
            density = jnp.atleast_1d(jnp.asarray(density))
            density = density / density.sum()
            self.density = eqx.error_if(
                density,
                density.shape != self.wavelength.shape,
                f"Density shape {density.shape} must match wavelength shape {self.wavelength.shape}.",
            )

    @classmethod
    def build(
        cls,
        spectrum: Spectrum
        | ScalarLike
        | Float[Array, "wv"]
        | tuple[Float[Array, "wv"], Float[Array, "wv"]],
    ) -> Spectrum | MonoSpectrum:
        """
        Function that creates a spectrum type from either a single float
        (MonoSpectrum), an array of multiple wavelengths, or a tuple of two
        arrays: one array of wavelengths and another array of the relative power
        of each wavelength in the spectrum.

        Args:
            spectrum: If this is a scalar float value, a single wavelength
                ``MonoSpectrum`` will be created. If this is a single array
                of floats, a ``Spectrum`` of multiple wavelengths that each
                have equivalent intensity will be created. If this is a tuple,
                the first element of the tuple will be used as the array of
                wavelengths and the second element of the tuple will be used
                as the array of weights representing the relative power of each
                wavelength in the spectrum.
        Returns:
            An object representing a spectrum (or a single wavelength in the
            monochrome case).
        """
        if isinstance(spectrum, Spectrum):
            return spectrum
        elif isinstance(spectrum, float) or isinstance(spectrum, int):
            return MonoSpectrum(wavelength=spectrum)
        elif isinstance(spectrum, tuple) or isinstance(spectrum, list):
            return Spectrum(wavelength=spectrum[0], density=spectrum[1])
        elif isinstance(spectrum, ArrayLike) and spectrum.size == 1:
            return MonoSpectrum(wavelength=spectrum.squeeze())
        else:
            return Spectrum(wavelength=spectrum)

    @property
    def size(self) -> int:
        return self.wavelength.size

    @property
    def central_wavelength(self) -> float:
        """
        The central wavelength of the spectrum (defined as the first wavelength
        provided to the spectrum because you could construct the complex field
        with multiple wavelengths in any order).
        """
        return self.wavelength[0]

    @property
    def spectral_modulation(self) -> Array:
        return self.central_wavelength / self.wavelength

    def __repr__(self) -> str:
        return eqx.tree_pformat(self, short_arrays=False)


class MonoSpectrum(Spectrum):
    """A spectrum with a scalar wavelength (density is a delta function)."""

    wavelength: Float[Array, "1"]
    density: Float[Array, "1"]

    def __init__(self, wavelength: float | Array):
        self.wavelength = jnp.atleast_1d(jnp.asarray(wavelength))
        assert self.wavelength.shape == (1,), (
            f"MonoSpectrum requires a scalar wavelength, got array of shape {self.wavelength.shape}"
        )
        self.density = jnp.ones((1,))
