from chromatix.elements import PhaseMask
from chromatix.functional import plane_wave
from jax.random import PRNGKey
import jax.numpy as jnp
from chromatix.elements.utils import trainable

key = PRNGKey(42)


def mask_init(key, field):
    return jnp.ones_like(field.u)


field = plane_wave((512, 512), 1.0, 0.532, 1.0)
model = PhaseMask(trainable(mask_init))
params = model.init(key, field)
print(params.keys())
# params.pop("params")
