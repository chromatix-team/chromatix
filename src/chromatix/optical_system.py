from flax import linen as nn
from jax import vmap

from .ops import fourier_convolution

from .field import Field
from chex import Array, PRNGKey
from typing import Callable, Sequence, Optional, Any


class Microscope(nn.Module):
    optical_system: Sequence[Callable]
    noise_fn: Optional[Callable[[PRNGKey, Array], Array]] = None
    reduce_fn: Optional[Callable] = None

    def setup(self):
        self.psf_model = OpticalSystem(self.optical_system)

    def __call__(self, data: Array, *args: Any, **kwargs: Any) -> Array:
        psf = self.psf(*args, **kwargs)
        return self.image(psf, data)

    def psf(self, *args: Any, **kwargs: Any) -> Array:
        return self.psf_model(*args, **kwargs).intensity

    def output_field(self, *args: Any, **kwargs: Any) -> Field:
        return self.psf_model(*args, **kwargs)

    def image(self, psf: Array, data: Array) -> Array:
        image = vmap(fourier_convolution, in_axes=(0, 0))(data, psf)
        if self.reduce_fn is not None:
            image = self.reduce_fn(image)
        if self.noise_fn is not None:
            image = self.noise_fn(self.make_rng("noise"), image)
        return image


class OpticalSystem(nn.Module):
    elements: Sequence[Callable]

    @nn.compact
    def __call__(self, *args: Any, **kwargs: Any) -> Field:
        field = self.elements[0](*args, **kwargs)  # allow field to be initialized
        for element in self.elements[1:]:
            field = element(field)
        return field
