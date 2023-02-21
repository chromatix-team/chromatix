import jax.numpy as jnp

from ..field import Field
from ..functional.phase_masks import wrap_phase, phase_change
from typing import Callable, Union, Tuple
from einops import rearrange
from flax import linen as nn
from chex import Array, PRNGKey, assert_rank
from jax.scipy.ndimage import map_coordinates

__all__ = ["PhaseMask", "SpatialLightModulator"]


class PhaseMask(nn.Module):
    phase: Union[Array, Callable[[PRNGKey, Tuple[int, ...]], Array]]

    @nn.compact
    def __call__(self, field: Field) -> Field:
        phase = (
            self.param("phase_pixels", self.phase, field.shape[1:3])
            if callable(self.phase)
            else self.phase
        )
        assert_rank(phase, 4, custom_message="Phase must be array of shape [1 H W 1]")
        phase = self.spectrally_modulate_phase(
            phase, field.spectrum, field.spectrum[0].item()
        )
        return phase_change(field, phase)


class SpatialLightModulator(nn.Module):
    phase: Union[Array, Callable[[PRNGKey, Tuple[int, ...]], Array]]
    shape: Tuple[int, int]
    phase_range: Tuple[float, float]
    interpolation_order: int = 0

    @nn.compact
    def __call__(self, field: Field) -> Field:
        phase = (
            self.param("slm_pixels", self.phase, self.shape)
            if callable(self.phase)
            else self.phase
        )
        assert_rank(phase, 4, custom_message="Phase must be array of shape [1 H W 1]")
        phase = wrap_phase(phase, self.phase_range)
        field_pixel_grid = jnp.meshgrid(
            jnp.linspace(0, self.shape[0] - 1, num=field.shape[1]) + 0.5,
            jnp.linspace(0, self.shape[1] - 1, num=field.shape[2]) + 0.5,
            indexing="ij",
        )
        phase = map_coordinates(
            phase.squeeze(), field_pixel_grid, self.interpolation_order
        )
        phase = rearrange(phase, "h w -> 1 h w 1")
        phase = self.spectrally_modulate_phase(
            phase, field.spectrum, field.spectrum[0].item()
        )
        return phase_change(field, phase)

    @staticmethod
    def spectrally_modulate_phase(
        phase: Array, spectrum: Array, central_wavelength: float
    ) -> Array:
        assert_rank(
            spectrum, 4, custom_message="Spectrum must be array of shape [1 1 1 C]"
        )

        spectral_modulation = central_wavelength / spectrum
        return phase * spectral_modulation
