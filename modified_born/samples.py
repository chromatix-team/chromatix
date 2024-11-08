import jax.numpy as jnp
from einops import reduce
from jax import Array


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


def angled_interface():
    return (
        jnp.full((1000, 1000), 1.0)
        .at[jnp.triu_indices(n=1000)]
        .set(1.55)[::-1, None, :]
    )
