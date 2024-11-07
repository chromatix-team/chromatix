from __future__ import annotations

from functools import reduce as freduce
from typing import Literal

import jax.numpy as jnp
import numpy as np
from chex import dataclass
from einops import reduce
from jax import Array
from scipy.ndimage import distance_transform_edt
from scipy.special import factorial


@dataclass
class Source:
    field: Array
    wavelength: float

    @property
    def source(self):
        return (2 * jnp.pi / self.wavelength) ** 2 * self.field


@dataclass
class Sample:
    # Simple dataclass to deal with refractive index
    # padding, boundary conditions etc. Nothing vital
    # but makes life a lot easier.
    permittivity: Array
    dx: float
    roi: tuple[slice, ...]

    @classmethod
    def init(
        cls,
        refractive_index: Array,  # [N_z, N_y, N_x]
        dx: float,  # in wavelengths
        wavelength: float,
        boundary_width: tuple[None | int, ...],
        boundary_type: Literal["pbl", "arl"] = "arl",
        *,
        boundary_strength: float = 0.2,
        boundary_order: int = 4,
    ) -> Sample:
        permittivity, roi = cls.add_boundary(
            refractive_index**2,
            wavelength,
            dx,
            boundary_type,
            boundary_width,
            boundary_order,
            boundary_strength,
        )
        permittivity = cls.pad_fourier(permittivity, boundary_width)

        return cls(
            permittivity=permittivity,
            dx=dx,
            roi=roi,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.permittivity.shape

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.permittivity.shape[:3]

    @property
    def k_grid(self) -> Array:
        # NOTE THE 2PI factor!!!
        ks = [
            2 * jnp.pi * jnp.fft.fftfreq(shape, self.dx) for shape in self.spatial_shape
        ]
        return jnp.stack(jnp.meshgrid(*ks, indexing="ij"), axis=-1)

    @staticmethod
    def add_boundary(
        permittivity: Array,
        wavelength: float,
        dx: float,
        type: Literal["arl", "pbl"],
        width: tuple[None | int, ...],
        order: int,
        strength: float,
    ) -> tuple[Array, tuple[slice, ...]]:
        # Finding new shapes and rois
        n_pad = tuple(0 if width_i is None else int(width_i / dx) for width_i in width)
        roi = tuple(slice(n, n + size) for n, size in zip(n_pad, permittivity.shape))

        # Padding permittivity to new size
        padding = [(0, 0) for _ in range(permittivity.ndim)]
        for idx, n in enumerate(n_pad):
            padding[idx] = (n, n)
        permittivity = jnp.pad(permittivity, padding, mode="edge")

        # Adding BCs
        match type:
            case "pbl":
                permittivity = Sample.add_pbl(
                    permittivity, roi, strength, order, wavelength, dx
                )
            case "arl":
                permittivity = Sample.add_arl(permittivity, roi, np.max(n_pad))
        return permittivity, roi

    @staticmethod
    def add_pbl(
        permittivity: Array,
        roi: tuple[slice, ...],
        strength: float,
        order: int,
        wavelength: float,
        dx: float,
    ) -> Array:
        # Gathering constants
        km = 2 * jnp.pi * jnp.sqrt(jnp.mean(permittivity)) * dx / wavelength
        alpha = strength * km**2 / (2 * km)

        # Defining distance from sample
        r = jnp.ones_like(permittivity).at[roi].set(0)
        r = distance_transform_edt(r)

        # Making boundary
        ar = alpha * r
        P = freduce(
            lambda P, n: P + (ar**n / factorial(n, exact=True)),
            range(order + 1),
            jnp.zeros_like(ar),
        )

        numerator = alpha**2 * (order - ar + 2 * 1j * km * r) * ar ** (order - 1)
        denominator = P * factorial(order, exact=True)
        boundary = 1 / km**2 * numerator / denominator

        return permittivity + boundary

    @staticmethod
    def add_arl(permittivity: Array, roi: tuple[slice, ...], n_pad: Array) -> Array:
        # Defining distance from sample
        r = jnp.ones_like(permittivity).at[roi].set(0)
        r = distance_transform_edt(r)
        beta = (jnp.abs(n_pad - r) - 0.21) / (n_pad + 0.66)
        beta = beta.at[roi].set(1.0)
        return permittivity * beta

    @staticmethod
    def pad_fourier(x: Array, width: tuple[int | None, ...]) -> Array:
        # Pads to fourier friendly shapes (powers of 2), depending
        # on periodic or absorbing BCs
        def n_pad(order, size):
            padding = 0 if order is None else int(2 ** np.ceil(np.log2(size))) - size
            return (0, padding)

        padding = [(0, 0) for _ in range(x.ndim)]
        for idx, (order, size) in enumerate(
            zip(width, x.shape[: len(width)], strict=True)
        ):
            padding[idx] = n_pad(order, size)
        return jnp.pad(x, padding, mode="constant", constant_values=0)


def sample_grid(size: tuple[int, int, int]) -> Array:
    N_z, N_y, N_x = size

    grid = jnp.meshgrid(
        jnp.linspace(-N_z // 2, N_z // 2 - 1, num=N_z) + 0.5,
        jnp.linspace(N_y // 2, -N_y // 2 - 1, num=N_y) + 0.5,
        jnp.linspace(-N_x // 2, N_x // 2 - 1, num=N_x) + 0.5,
        indexing="ij",
    )

    return jnp.stack(grid, axis=-1)


def cylinders(
    size: tuple[int, int, int],
    spacing: float,
    location: Array,
    radius: float,
    n_background: float,
    n_cylinder: float,
    antialiasing: int | None = 10,
) -> Array:
    # Making the grid, in 2D
    N_z, N_y, N_x = size

    if antialiasing is not None:
        N_z, N_x = N_z * antialiasing, N_x * antialiasing
        spacing = spacing / antialiasing

    grid = spacing * sample_grid((N_z, 1, N_x)).squeeze(1)[..., [0, 2]]

    # Making mask
    mask = jnp.zeros((N_z, N_x))
    for cylinder in location:
        mask += jnp.linalg.norm(grid - cylinder, axis=-1) < radius

    sample = jnp.where(mask == 1.0, n_cylinder, n_background)
    if antialiasing is not None:
        sample = reduce(sample, f"(z {antialiasing}) (x {antialiasing}) -> z x", "mean")

    # Setting index
    sample = jnp.repeat(sample[:, None, :], N_y, axis=1)
    return sample


def vacuum_cylinders():
    cylinder_locs = jnp.array(
        [[-44.5, -44.5], [44.5, -44.5], [44.5, 44.5], [20.6, -18.0], [-10.4, 18.1]]
    )

    cylinder_radius = 5.0
    n_cylinder = 1.2
    n_mat = 1.0
    spacing = 0.1
    sim_size = 100
    sim_shape = int(sim_size / spacing)
    shape = (sim_shape, 1, sim_shape)

    return cylinders(
        shape,
        spacing,
        cylinder_locs,
        cylinder_radius,
        n_mat,
        n_cylinder,
        antialiasing=10,
    )


def bio_cylinders():
    cylinder_locs = jnp.array(
        [[-44.5, -44.5], [44.5, -44.5], [44.5, 44.5], [20.6, -18.0], [-10.4, 18.1]]
    )

    cylinder_radius = 5.0
    n_cylinder = 1.36
    n_mat = 1.33
    spacing = 0.1
    sim_size = 100
    sim_shape = int(sim_size / spacing)
    shape = (sim_shape, 1, sim_shape)

    return cylinders(
        shape,
        spacing,
        cylinder_locs,
        cylinder_radius,
        n_mat,
        n_cylinder,
        antialiasing=10,
    )
