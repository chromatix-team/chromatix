import jax.numpy as jnp
from chromatix.ops.resample import init_plane_resample

def test_resampling():
    grid = jnp.stack(jnp.meshgrid(jnp.arange(9), jnp.arange(9)), axis=-1)
    vals = jnp.max(grid, axis=-1)
    resampled = init_plane_resample((3, 3), (1, 1))(vals[..., None], (1, 1))

    # Grid should be aligned so sum of values should be 3 + 3 * 4 + 5 * 5 etc
    assert jnp.allclose(jnp.sum(resampled), 40)