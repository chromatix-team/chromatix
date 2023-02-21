import jax.numpy as jnp

from ..field import Field
from ..functional.phase_masks import wrap_phase, phase_change
from typing import Callable, Union, Tuple
from flax import linen as nn
from chex import Array, PRNGKey, assert_rank
from jax.scipy.ndimage import map_coordinates

__all__ = ["PhaseMask", "SpatialLightModulator"]


class PhaseMask(nn.Module):
    phase: Union[Array, Callable[[PRNGKey, Tuple[int, ...]], Array]]

    @nn.compact
    def __call__(self, field: Field) -> Field:
        _phase = (
            self.param("_phase", self.phase, (1, *field.shape[1:3], 1))
            if callable(self.phase)
            else self.phase
        )
        assert_rank(_phase, 4, custom_message="Phase must have shape [N, H, W, C]")
        return field * jnp.exp(1j * _phase)


class SpatialLightModulator(nn.Module):
    phase_init_fn: Callable[[PRNGKey], Array]
    spacing: Tuple[float, float]
    phase_range: Tuple[float, float] = (1.4 * jnp.pi, 4.6 * jnp.pi)
    interpolation_order: int = 0

    @nn.compact
    def __call__(self, field: Field) -> Field:
        slm_pixels = self.param("phase", self.phase_init_fn)
        field_grid = jnp.meshgrid(
            jnp.linspace(0, self.shape[0] - 1, num=field.shape[1]) + 0.5,
            jnp.linspace(0, self.shape[1] - 1, num=field.shape[2]) + 0.5,
            indexing="ij",
        )
        phase = map_coordinates(slm_pixels, field_grid, self.interpolation_order)
        phase = wrap_phase(phase, self.phase_range)
        phase = self.spectrally_modulate_phase(
            phase, field.spectrum, field.spectrum[0].item()
        )
        phase = wrap_phase(phase, self.phase_range)
        return phase_change(field, jnp.exp(1j * phase[None, ..., None]))

    @staticmethod
    def spectrally_modulate_phase(
        phase: Array, spectrum: Array, central_wavelength: float
    ) -> Array:
        assert_rank(
            spectrum, 4, custom_message="Spectrum must be ndarray of shape [1 1 1 c]"
        )

        spectral_modulation = central_wavelength / spectrum
        return phase * spectral_modulation
