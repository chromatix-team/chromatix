from chromatix.elements import ObjectivePointSource, PhaseMask, FFLens
from chromatix import PSFImager
from chromatix.ops.noise import approximate_shot_noise
import jax.numpy as jnp
from jax import jit
from jax import random
import flax.linen as nn
from typing import Callable, Optional, Tuple
from chromatix.utils import center_crop
from chromatix.ops.ops import downsample
from chex import Array
import numpy as np
from chromatix.ops.windows import sigmoid_taper
from chromatix import Field

key = random.PRNGKey(42)


class Holoscope(nn.Module):
    camera_shape: Tuple[int, int] = (512, 512)
    camera_pixel_pitch: float = 0.325
    f: float = 100
    NA: float = 0.8
    n: float = 1.33
    wavelength: float = 0.532
    wavelength_ratio: float = 1.0
    upsample: int = 2
    pad: int = 512
    taper_width: Optional[float] = None

    noise_fn: Callable = approximate_shot_noise

    def setup(self):
        # Getting shapes and spacing right
        assert isinstance(self.upsample, int), "Upsample factor must be integer."
        assert isinstance(self.pad, int), "Padding must be integer."
        shape = tuple(np.array(self.camera_shape) * self.upsample + self.pad)
        # Spacing in SLM plane should be upsample * f * wavelength / (n * N_field * camera_spacing)
        spacing = (
            self.upsample
            * self.f
            * self.wavelength
            / (self.n * shape[0] * self.camera_pixel_pitch)
        )

        self.model = PSFImager(
            [
                ObjectivePointSource(
                    shape,
                    spacing,
                    self.wavelength,
                    self.wavelength_ratio,
                    self.f,
                    self.n,
                    self.NA,
                ),
                PhaseMask(random.normal),
                FFLens(self.f, self.n),
            ],
            noise_fn=self.noise_fn,
            reduce_fn=lambda x: jnp.sum(x, axis=0, keepdims=True),
        )
        if self.taper_width is not None:
            self.taper = sigmoid_taper(self.camera_shape, self.taper_width)

    def __call__(self, data: Array, z: Array):
        # Post processing of psf
        psf = self.psf(z, post_process=True)
        return self.model.image(psf, data)

    def psf(self, z: Array, post_process: bool = True) -> Array:
        psf = self.model.psf(z)
        if post_process:
            psf = center_crop(psf, (None, self.pad // 2, self.pad // 2, None))
            psf = downsample(psf, (self.upsample, self.upsample))
            if self.taper_width is not None:
                psf = psf * self.taper
        return psf

    def output_field(self, z: Array) -> Field:
        return self.model.output_field(z)


z = jnp.linspace(-150, 150, 5)
camera_shape = (512, 512)

data = random.normal(key, (z.size, *camera_shape, 1))
model = Holoscope(camera_shape, noise_fn=None)
params = model.init({"params": key, "noise": key}, data, z)
assert model.apply(params, z, method=model.output_field).dx.squeeze() == 0.325 / 2

forward = jit(model.apply)
print(forward(params, data, z, rngs={"noise": key}))
print(forward(params, data, z, rngs={"noise": key}).shape)
print(params)
