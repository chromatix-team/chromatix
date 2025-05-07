import jax.numpy as jnp


def universal_compensator_modes(setting: int = 0, swing: float = 0):
    """Settings for the LC-PolScope polarizer
    Parameters:
        setting (int): LC-PolScope setting number between 0 and 4
        swing (float): proportion of wavelength, for ex 0.03
    Returns:
        tuple: retardance values for the universal compensator
    """
    swing_rad = swing * 2 * jnp.pi
    match setting:
        case 0:
            retA = jnp.pi / 2
            retB = jnp.pi
        case 1:
            retA = jnp.pi / 2 + swing_rad
            retB = jnp.pi
        case 2:
            retA = jnp.pi / 2
            retB = jnp.pi + swing_rad
        case 3:
            retA = jnp.pi / 2
            retB = jnp.pi - swing_rad
        case 4:
            retA = jnp.pi / 2 - swing_rad
            retB = jnp.pi
        case _:
            raise NotImplementedError
    return retA, retB
