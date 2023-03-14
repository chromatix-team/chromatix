import jax.numpy as jnp


def l2_sq_norm(grid: jnp.ndarray) -> jnp.ndarray:
    """Sum of the squared grid over spatial dimensions, i.e. `x**2 + y**2`."""
    return jnp.sum(grid**2, axis=0)


def l2_norm(grid: jnp.ndarray) -> jnp.ndarray:
    """Square root of ``l2_sq_norm``, i.e. `sqrt(x**2 + y**2)`."""
    return jnp.sqrt(jnp.sum(grid**2, axis=0))


def l1_norm(grid: jnp.ndarray) -> jnp.ndarray:
    """Sum absolute value over spatial dimensions, i.e. `|x| + |y|`."""
    return jnp.sum(jnp.abs(grid), axis=0)


def linf_norm(grid: jnp.ndarray) -> jnp.ndarray:
    """Max absolute value over spatial dimensions, i.e. `max(|x|, |y|)`."""
    return jnp.max(jnp.abs(grid), axis=0)
