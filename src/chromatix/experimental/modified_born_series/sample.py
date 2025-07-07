from typing import Self

import equinox as eqx
import jax.numpy as jnp
import scipy.constants as const
from jax import Array


class Sample(eqx.Module):
    permittivity: Array
    spacing: float
    ROI: tuple[slice, slice, slice]

    def __init__(self, permittivity: Array, spacing: float):
        # Add singleton dimensions on the left.
        self.permittivity = permittivity

        # Spacing
        self.spacing = spacing

        self.ROI = (
            slice(0, self.shape[0]),
            slice(0, self.shape[1]),
            slice(0, self.shape[2]),
        )

    @property
    def shape(self):
        return self.permittivity.shape

    @property
    def grid(self) -> Array:
        N_x, N_y, N_z = self.shape
        offset = [N // 2 if N != 1 else 0 for N in self.shape]
        grid = jnp.stack(
            jnp.meshgrid(
                jnp.linspace(0, (N_x - 1), N_x) - offset[0],
                jnp.linspace(0, (N_y - 1), N_y) - offset[1],
                jnp.linspace(0, (N_z - 1), N_z) - offset[2],
                indexing="ij",
            ),
            axis=-1,
        )
        return grid * self.spacing

    @property
    def k_grid(self) -> Array:
        N_x, N_y, N_z = self.shape
        grid = jnp.stack(
            jnp.meshgrid(
                jnp.fft.fftfreq(N_x),
                jnp.fft.fftfreq(N_y),
                jnp.fft.fftfreq(N_z),
                indexing="ij",
            ),
            axis=-1,
        )
        return 2 * jnp.pi * grid / self.spacing

    @property
    def extent(self) -> Array:
        return jnp.asarray(self.shape) * self.spacing

    def replace(self, **kwargs) -> Self:
        for key, value in kwargs.items():
            where_fn = lambda tree: getattr(tree, key)  # noqa: E731
            self = eqx.tree_at(where_fn, self, value)
        return self


def EmptySample(shape: tuple[int, int, int], spacing: float) -> Sample:
    return Sample(jnp.zeros(shape), spacing)


class Source(eqx.Module):
    k0: float | Array
    field: Array

    def __init__(self, current_density: Array, k0: float | Array):
        self.k0 = k0
        self.field = -1j / k0 * 1e-6 * const.c * const.mu_0 * current_density

    @property
    def current_density(self):
        return self.field / (-1j / self.k0 * 1e-6 * const.c * const.mu_0)


def add_absorbing_bc(
    sample: Sample,
    axis: int | tuple[int],
    thickness: float,
    max_extinction: float = 0.25,
    n_background: float = 1.0,
) -> Sample:
    if not isinstance(axis, tuple):
        axis = (axis,)

    boundary_width = int(thickness / sample.spacing)
    boundary_shape = tuple(
        (boundary_width, boundary_width) if idx in axis else (0, 0) for idx in range(3)
    )
    roi = tuple(
        slice(width, width + shape)
        for (width, _), shape in zip(boundary_shape, sample.shape)
    )

    extinction = jnp.pad(
        jnp.zeros(sample.shape),
        boundary_shape,
        mode="linear_ramp",
        end_values=max_extinction,
    )

    epsilon = (n_background + 1j * extinction) ** 2
    permittivity = jnp.pad(
        sample.permittivity,
        boundary_shape,
        mode="constant",
        constant_values=n_background**2,
    )

    return sample.replace(
        permittivity=permittivity + (epsilon - n_background**2), ROI=roi
    )
