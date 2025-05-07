# %% Imports
# Imports
import os
import jax
import jax.numpy as jnp
from jax import lax
from flax.struct import PyTreeNode
from dataclasses import replace
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Chromatix imports
from chromatix import VectorField
import chromatix.functional as cf
from chromatix.ops import init_plane_resample

# Local module imports
from sample import single_bead_sample
from tensor_tomo import thick_polarised_sample
from helpers import (
    apply_jones_in_lab_basis, apply_jones_in_wave_basis,
    compute_angles_from_offset, plot_camera_images
)
from helpers_sampling import (
    fibonacci_cone_sampling, generate_kohler_2d_points
)


# %% Helper functions
# Helper functions
def incoherent_sum_across_angles(final_amps: jnp.ndarray, normalize: bool = False) -> jnp.ndarray:
    """
    Incoherently sums intensity across the first dimension of final_amps.
    final_amps shape: (num_angles, ..., 3).
    """
    if isinstance(final_amps, list):
        final_amps = jnp.array(final_amps)
    amps_sq = jnp.abs(final_amps) ** 2
    intensities_sum = jnp.sum(amps_sq, axis=0)
    if normalize:
        total_power = jnp.sum(intensities_sum)
        intensities_sum /= (total_power + 1e-16)
    return intensities_sum

def universal_compensator_modes(swing: float) -> jnp.ndarray:
    """
    Returns a set of universal compensator modes with relative phase adjustments.
    """
    uc_modes = jnp.array([
        [jnp.pi / 2, jnp.pi],
        [jnp.pi / 2 + swing, jnp.pi],
        [jnp.pi / 2, jnp.pi + swing],
        [jnp.pi / 2, jnp.pi - swing],
        [jnp.pi / 2 - swing, jnp.pi],
    ])
    # Create a dummy field to apply compensator for phase extraction
    field = cf.plane_wave((1, 1), 1, 1, amplitude=cf.linear(0.0))
    field = jax.vmap(cf.universal_compensator, in_axes=(None, 0, 0))(
        field, uc_modes[:, 0], uc_modes[:, 1]
    )
    amplitudes = field.u.squeeze()
    phase_adjust = jnp.exp(1j * jnp.angle(amplitudes[:, 1]))
    amplitudes = amplitudes * phase_adjust[:, None] * 2.0
    return amplitudes

def gaussian_spot_field(
    shape: tuple[int, int],
    dx: float,
    spectrum: float,
    spectral_density: float,
    waist: float,
    center: tuple[float, float] = (0.0, 0.0),
    polarization: jnp.ndarray | None = None,
    amplitude: float = 1.0
) -> VectorField:
    """
    Creates a Gaussian spot (transverse) in a VectorField of shape (1, H, W, 1, 3).
    """
    field = VectorField.create(dx, spectrum, spectral_density, shape=shape)
    if polarization is None:
        polarization = jnp.array([1.0, 0.0, 0.0], dtype=field.u.dtype)

    Y, X = field.grid[0, 0, ..., 0, 0], field.grid[1, 0, ..., 0, 0]
    envelope = jnp.exp(-((X - center[0])**2 + (Y - center[1])**2) / (waist**2))
    envelope_expanded = envelope[None, :, :, None]
    u = amplitude * envelope_expanded[..., None] * polarization
    field = field.replace(u=u)
    return field


# %% PolScope Class Definition
class PolScope(PyTreeNode):
    """
    Simplified PolScope pipeline:
      - Condenser lens (ff_lens_debye_chunked)
      - Sample interaction (thick_polarised_sample)
      - Analyzer (apply_jones_in_wave_basis)
      - Microlens array (rectangular_microlens_array)
      - Transfer propagate
      - Resample to camera
    """
    # Grid & physical params
    shape: tuple[int, int] = (128, 128)
    spacing: float = 0.546 / 2
    wavelength: float = 0.546
    swing: float = 2 * jnp.pi * 0.03
    num_angles: int = 3

    # Condenser lens
    condenser_f: float = 5000 
    condenser_n: float = 1.33
    condenser_NA: float = 0.8

    # Objective
    objective_f: float = 5000
    objective_n: float = 1.33
    objective_NA: float = 0.8

    # Tube lens
    tube_f: float = 200_000
    tube_n: float = 1.0
    tube_NA: float = 1.0

    # Microlens array (MLA)
    mla_n: float = 1.0
    mla_f: float = 2500
    mla_radius: float = 50
    mla_separation: float = 100
    mla_n_y: float = 9
    mla_n_x: float = 9

    # Camera
    camera_shape: tuple[int, int] = (144, 144)
    camera_pitch: float = 6.5

    @property
    def angles(self) -> jnp.ndarray:
        return fibonacci_cone_sampling(
            num_angles=self.num_angles,
            NA=self.condenser_NA,
            n=self.condenser_n,
            overshoot_factor=10
        )

    @property
    def kohler_points(self) -> jnp.ndarray:
        """
        2D coordinates for illumination in the condenser pupil plane,
        scaled to physical units in microns.
        """
        points = generate_kohler_2d_points(
            num_angles=self.num_angles,
            NA=self.condenser_NA,
            n=self.condenser_n
        )
        lens_radius = self.condenser_f * jnp.tan(jnp.arcsin(self.condenser_NA / self.condenser_n))
        normalized_max = self.condenser_NA / self.condenser_n
        physical_scaling = lens_radius / normalized_max
        y_microns = points[:, 0] * physical_scaling
        x_microns = points[:, 1] * physical_scaling
        scaled_points = jnp.column_stack((y_microns, x_microns))
        return scaled_points

    def __call__(self, potential: jnp.ndarray):
        """
        Run the PolScope for each universal compensator mode, returning the 5-camera images.
        """
        uc_modes = universal_compensator_modes(self.swing)
        # vmapped over each UC mode => returns shape (5, camera_shape)
        return jax.vmap(self.forward_chunked, in_axes=(0, None))(uc_modes, potential)

    def single_angle_forward(self, uc_mode: jnp.ndarray, potential: jnp.ndarray, kohler_pt: jnp.ndarray):
        """
        Forward model for a single angle.
        """
        # 1) Illumination
        field = gaussian_spot_field(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            spectral_density=1.0,
            waist=1.0 / 4,
            center=(kohler_pt[0], kohler_pt[1]),
            polarization=uc_mode * 1000.0
        )

        # 2) Condenser lens
        field = cf.ff_lens_debye_chunked(
            field,
            self.condenser_f,
            self.condenser_n,
            self.condenser_NA,
            range_um=(field.u.shape[2] * field.dx.squeeze()[1]) * 2,
            num_samples=potential.shape[1],
            chunk_size=1024
        )

        # 3) Sample interaction
        dz = field.dx.squeeze()[0] * 8
        field = thick_polarised_sample(
            field,
            potential,
            self.objective_n,
            dz,
            NA=self.objective_NA
        )

        # 4) Propagate half sample thickness (optional extra propagation)
        z_sample = potential.shape[0] * dz
        field = cf.transfer_propagate(field, -z_sample / 2, self.objective_n, 512, mode="same")

        # 5) Analyzer
        M_wave_2x2 = jnp.array([
            [1.0/jnp.sqrt(2),  1j/jnp.sqrt(2)],
            [-1j/jnp.sqrt(2),  1.0/jnp.sqrt(2)]
        ], dtype=jnp.complex64)
        angle = compute_angles_from_offset(kohler_pt[1], kohler_pt[0], self.objective_f)
        field = apply_jones_in_wave_basis(
            field,
            M_wave_2x2.T,
            n_medium=self.objective_n,
            spectrum=self.wavelength,
            angle=angle
        )

        # 6) Microlens array
        mla_radius = jnp.floor(self.mla_radius / field.dx.squeeze()[0]) * field.dx.squeeze()[0]
        mla_separation = jnp.floor(self.mla_separation / field.dx.squeeze()[0]) * field.dx.squeeze()[0]
        field = cf.rectangular_microlens_array(
            field,
            n=self.mla_n,
            f=self.mla_f,
            num_lenses_height=self.mla_n_y,
            num_lenses_width=self.mla_n_x,
            radius=mla_radius,
            separation=mla_separation,
            block_between=True
        )

        # 7) Transfer propagate after MLA
        field = cf.transfer_propagate(field, self.mla_f, self.mla_n, 512, mode="same")
        return field

    def forward_chunked(self, uc_mode: jnp.ndarray, potential: jnp.ndarray, chunk_size: int = 5):
        """
        Forward model for all angles (in chunks), then sum incoherently.
        Returns (incoherent_field, camera_image).
        """
        def per_angle(a):
            return self.single_angle_forward(uc_mode, potential, a).u

        # Gather final amplitudes from each angle
        kohler_pts = self.kohler_points
        angle_batches = []
        for i in range(0, len(kohler_pts), chunk_size):
            chunk = kohler_pts[i:i+chunk_size]
            chunk_amps = jax.vmap(per_angle, in_axes=0)(chunk)
            angle_batches.append(chunk_amps)

        final_amps = jnp.concatenate(angle_batches, axis=0)
        I_incoh = incoherent_sum_across_angles(final_amps, normalize=False)

        # Construct a field for the incoherent sum
        ref_field = VectorField.create(self.spacing, self.wavelength, 1, shape=self.shape)
        u_incoh = jnp.sqrt(I_incoh) * jnp.exp(1j * 0.0)
        incoherent_field = ref_field.replace(u=u_incoh)

        # Downsample/Resample to camera
        camera = init_plane_resample(self.camera_shape, self.camera_pitch)(
            incoherent_field.intensity.squeeze(0),
            incoherent_field.dx.squeeze()
        )
        return incoherent_field, camera.squeeze()


# %% Main Simulation
# Main Simulation

# --- 1) Define the sample to simulate ---
# Here we're creating a 3D potential (birefringent bead) on a 1024×1024 grid,
# with 3 slices in z, and an overall refractive-index background of 1.33.
sample_pixels = 1024
potential_z = single_bead_sample(
    n_background=1.33,
    n_bead=jnp.array([1.45, 1.33, 1.33]),   # Birefringent bead: nz=1.45, ny=1.33, nx=1.33
    orientation=jnp.array([0, 0, 0]),      # No tilt in its principal axes
    radius=7.5,                            # Bead radius in microns
    shape=(3, sample_pixels, sample_pixels),
    spacing=0.546 / 4,                     # Spatial sampling in microns
    k0=2 * jnp.pi / 0.546,                 # Wave number (2π / wavelength[µm])
)[:-1, :-1, :-1, None]                     # Slice to avoid boundary effects & add channel dim

# --- 2) Create a base PolScope instance ---
scope = PolScope()

# --- 3) Configure geometry and scaling for a shorter scope ---
#    We'll shrink the scope's grid so that the physical pupil diameter 
#    accommodates the condenser's NA at focal_length=40 µm.
focal_length = 40
NA = scope.condenser_NA
n = scope.condenser_n

# Calculate the required diameter (in microns) for the condenser pupil
diam_um = 2 * focal_length * jnp.tan(jnp.arcsin(NA / n))

# Set up a grid that definitely fits the pupil
res_factor = 2
shape_len = 128 * res_factor * 2
spacing = scope.wavelength / 2 / res_factor
pupil_field_width_um = shape_len * spacing

# Check we have enough grid size to capture the lens pupil
assert pupil_field_width_um >= diam_um, (
    f"Grid too small: needed >= {diam_um:.2f} µm, got {pupil_field_width_um:.2f} µm"
)

# --- 4) Adjust microlens array (MLA) parameters & camera ---
mag = 20
mla_f = scope.mla_f / mag
mla_f = 2  # override to be proportional to the focal length
mla_radius = scope.mla_radius / mag
mla_separation = scope.mla_separation / mag

# Create the MLA with 3 lenslets in x and y
mla_n_y = 3
mla_n_x = 3

# Camera sampling based on MLA size vs. desired pitch
camera_pitch = spacing
camera_pixels = int(jnp.ceil(mla_n_x * mla_radius * 2 / camera_pitch))

# --- 5) Create a modified (short) scope setup via dataclasses.replace ---
scope_short = replace(
    scope,
    num_angles=5,                     # reduce number of illumination angles for testing
    shape=(shape_len, shape_len),     # updated grid shape
    condenser_f=focal_length,
    objective_f=focal_length,
    condenser_NA=0.8,                 # override default
    spacing=spacing,
    tube_f=focal_length,
    mla_f=mla_f,
    mla_radius=mla_radius,
    mla_separation=mla_separation,
    mla_n_y=mla_n_y,
    mla_n_x=mla_n_x,
    camera_shape=(camera_pixels, camera_pixels),
    camera_pitch=camera_pitch,
)

# --- 6) Run the simulation for the first polarizer mode ---
field, camera = scope_short_angles.forward_chunked(uc_mode, potential_z)
plt.imshow(camera.squeeze(), cmap='inferno')
plt.colorbar()
plt.title(f"Camera Field")
plt.show()

# --- 7) Run the simulation (forward pass) ---
# Returns tuple: (incoherent_field, camera_image) for each UC mode
_, images = scope_short(potential_z)

# --- 8) Visualize the resulting camera images ---
fig = plot_camera_images(images)
