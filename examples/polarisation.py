import chromatix.functional as cf
import jax.numpy as jnp
from functools import partial

field = cf.plane_wave(
    (512, 512),
    1.0,
    0.532,
    1.0,
    amplitude=cf.linear(1 / 2 * jnp.pi),
    pupil=partial(cf.square_pupil, w=300.0),
    scalar=False,
)
print(field.shape)
field = cf.linear_polarizer(field, angle=1 / 4 * jnp.pi)
print(field.shape)
