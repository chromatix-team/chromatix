# %% [markdown]
# The code for 3D tomography for Aaron

# %%
from functools import partial
import chromatix.functional as cx
import chromatix.utils.fft as cfft
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import napari
from jax import Array
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

# %%
def create_volume(shape):
    return np.zeros(shape)

def random_location(shape, object_shape):
    h, w, l = shape
    oh, ow, ol = object_shape
    x = np.random.randint(0, h - oh)
    y = np.random.randint(0, w - ow)
    z = np.random.randint(0, l - ol)
    return x, y, z

def add_sphere(volume, radius):
    h, w, l = volume.shape
    x0, y0, z0 = random_location(volume.shape, (radius*2, radius*2, radius*2))
    for x in range(radius*2):
        for y in range(radius*2):
            for z in range(radius*2):
                if (x-radius)**2 + (y-radius)**2 + (z-radius)**2 <= radius**2:
                    volume[x0 + x, y0 + y, z0 + z] = 1
    return volume

def add_cube(volume, side_length):
    h, w, l = volume.shape
    x0, y0, z0 = random_location(volume.shape, (side_length, side_length, side_length))
    volume[x0:x0+side_length, y0:y0+side_length, z0:z0+side_length] = 2
    return volume

def add_pyramid(volume, base_length):
    h, w, l = volume.shape
    height = base_length // 2
    x0, y0, z0 = random_location(volume.shape, (base_length, base_length, height))
    for i in range(height):
        for x in range(base_length - 2*i):
            for y in range(base_length - 2*i):
                volume[x0 + x + i, y0 + y + i, z0 + i] = 3
    return volume

def plot_volume(volume):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of the voxels
    x, y, z = np.indices(volume.shape)

    # Extract the coordinates of the non-zero voxels
    x = x[volume > 0]
    y = y[volume > 0]
    z = z[volume > 0]
    values = volume[volume > 0]

    # Plot each object with different colors
    ax.scatter(x, y, z, c=values, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

# %%
shape = (16, 16, 16) # depth along the axis (depth), height, width
height, width, length = shape  # Adjust the size as needed
sample_phase = create_volume(shape)
sample_phase = add_sphere(sample_phase, 5)
sample_phase = add_cube(sample_phase, 5)
sample_phase = add_pyramid(sample_phase, 5)
sample_phase /= 6
# zero pad so that objects are in the center
padding = ((8, 8), (8, 8), (8, 8))
sample_phase = np.pad(sample_phase, padding, mode='constant', constant_values=0)

#%%
viewer, image_layer = napari.imshow(sample_phase, rgb=False)

# %%
#%% Global variables
key = jax.random.PRNGKey(42)
# thickness of each sample slice in microns
thickness = 1.0
# length of the field in pixels
size = sample_phase.shape[1:]
# pixel spacing
dx = 0.1154
# wavelength in microns
wavelength = 0.523
# power is 1 per pixel
power = np.prod(size) * dx * dx

def generate_hologram(sample_delay, sample_absorption):
    ref_field = cx.plane_wave(
        shape=size,
        dx=dx,
        spectrum=wavelength,
        spectral_density=1.0,
        power=power,
        kykx = (18/np.sqrt(2), 18/np.sqrt(2))
    )
    sample_field = cx.plane_wave(
        shape=size,
        dx=dx,
        spectrum=wavelength,
        spectral_density=1.0,
        power=power
    )
    sample_field = cx.multislice_thick_sample(
        sample_field,
        sample_absorption,
        sample_delay,
        1.33,
        thickness,
        N_pad=0,
    )
    return ref_field + sample_field

background_hologram = generate_hologram(sample_delay=np.zeros_like(sample_phase),
                            sample_absorption=np.zeros_like(sample_phase))
sample_hologram = generate_hologram(sample_delay=sample_phase,
                            sample_absorption=np.zeros_like(sample_phase))


fft_background = cfft.fft(background_hologram.intensity,
                        shift=True).squeeze()

fft_sample = cfft.fft(sample_hologram.intensity,
                        shift=True).squeeze()


# %%
def Ry(theta: float, dtype=jnp.float32) -> jnp.ndarray:
    """Generates rotation matrix around y.
    Theta in radians."""
    R = jnp.zeros((4, 4), dtype=dtype)
    sin_t = jnp.sin(theta).astype(dtype)
    cos_t = jnp.cos(theta).astype(dtype)

    R = R.at[1, 1].set(1.0)
    R = R.at[3, 3].set(1.0)  # homogeneous
    R = R.at[0, 0].set(cos_t)
    R = R.at[2, 2].set(cos_t)
    R = R.at[0, 2].set(sin_t)
    R = R.at[2, 0].set(-sin_t)

    return R

def volume_homogeneous_grid(volume: Array) -> Array:
    """Given a volume, generates a centred grid of homogeneous coordinates.
    Coordinates are placed along last dimension [z, y, x, 4]"""

    Nz, Ny, Nx = volume.shape
    z = jnp.linspace(-(Nz - 1) / 2, (Nz - 1) / 2, Nz, dtype=volume.dtype)
    y = jnp.linspace(-(Ny - 1) / 2, (Ny - 1) / 2, Ny, dtype=volume.dtype)
    x = jnp.linspace(-(Nx - 1) / 2, (Nx - 1) / 2, Nx, dtype=volume.dtype)
    grid = jnp.stack(jnp.meshgrid(z, y, x, indexing="ij"), axis=-1)
    return jnp.concatenate(
        [grid, jnp.ones((Nz, Ny, Nx, 1), dtype=volume.dtype)], axis=-1
    )

def resample(volume: Array, sample_grid: Array) -> Array:
    """Resample volume on coordinates given by grid.
    Assumes original coordinates were centered, i.e. -N/2 -> N/2"""
    offset = (jnp.array(volume.shape) - 1) / 2
    sample_locations = sample_grid.reshape(-1, 3).T + offset[:, None]
    resampled = map_coordinates(
        volume, list(sample_locations), order=1, mode="constant", cval=0.0
    )
    return resampled.reshape(sample_grid.shape[:3])

def rotate_volume(volume: Array, angle: float) -> Array:
    """Rotates a volume around the y axis (axis 1).
    angle in radians."""
    rotated_grid = volume_homogeneous_grid(volume) @ Ry(angle, volume.dtype).T
    return resample(volume, rotated_grid[..., :3])

# %%
rotated_vol = rotate_volume(sample_phase.astype(np.float32), np.pi/3)

# %%
viewer, image_layer = napari.imshow(rotated_vol, rgb=False)

# %%
plt.hist(rotated_vol.reshape(-1), 10)
plt.show()



# %%
