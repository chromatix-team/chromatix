import numpy as np
import jax.numpy as jnp
import os


def siemens_star(num_pixels=512, num_spokes=32):
    X, Y = np.mgrid[0:num_pixels, num_pixels:0:-1] - (num_pixels / 2.0)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(X, Y) + np.pi
    S = np.zeros_like(R)
    for spoke in range(num_spokes):
        in_spoke = (theta >= ((spoke) * 2 * np.pi / num_spokes)) & (
            theta <= ((spoke + 1) * 2 * np.pi / num_spokes)
        )
        if not spoke % 2:
            S[in_spoke] = 1.0
    S *= R < (num_pixels / 2.0)
    return S
