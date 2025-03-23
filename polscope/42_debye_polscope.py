# %%
%load_ext autoreload
%autoreload 2

import os
import sys
import numpy as np
import time
import jax
from jax import lax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from flax.struct import PyTreeNode
from dataclasses import replace
from tqdm.auto import tqdm

# Chromatix imports
from chromatix import VectorField
from chromatix.typing import ArrayLike, ScalarLike
import chromatix.functional as cf
from chromatix.ops import init_plane_resample

# Append parent directory to find local modules
sys.path.append("../")
from sample import single_bead_sample
from tensor_tomo import thick_polarised_sample

from helpers import (
    plot_propagated_field, plot_camera_images, plot_retardance_azimuth,
    ret_and_azim_from_intensity, ret_and_azim_from_intensity_with_bg,
    generate_mla_mask, plot_propagated_field_separate_colorbars,
    apply_jones_in_lab_basis, apply_jones_in_wave_basis,
    compute_angles_from_offset
)
from helpers_sampling import (
    fibonacci_cone_sampling, generate_kohler_2d_points
)
# from debye import optical_debye_wolf
from chromatix.functional.convenience import optical_debye_wolf
from debye import (
    optical_debye_wolf_factored_chunked
)

import importlib
importlib.reload(cf)
# Define directory for saving results
SAVE_DIR = os.path.join('results', 'debye_polscope_3')
os.makedirs(SAVE_DIR, exist_ok=True)


# %% Helper functions
def incoherent_sum_across_angles(final_amps: jnp.ndarray, normalize: bool = False) -> jnp.ndarray:
    """
    Incoherently sums intensity across the angle dimension of the vector field.

    Parameters
    ----------
    final_amps : jnp.ndarray
        The electric fields with shape (num_angles, ..., 3), where the last dimension
        of size 3 holds the vectorial components (E_x, E_y, E_z), and the first dimension
        is the illumination angle.
    normalize : bool, optional
        If True, normalize the total power of the returned array to 1.

    Returns
    -------
    jnp.ndarray
        The incoherently summed intensities across angles, preserving the vector dimension.
        If the input has shape (num_angles, 1, H, W, 1, 3), the returned shape will be
        (1, H, W, 1, 3).
    """
    # Compute |E|^2 for each angle without summing the vector dimension:
    if isinstance(final_amps, list):
        final_amps = jnp.array(final_amps)
    amps_sq = jnp.abs(final_amps) ** 2  # shape: (num_angles, ..., 3)

    # Sum over angles (axis=0), leaving vector components intact:
    intensities_sum = jnp.sum(amps_sq, axis=0)  # shape: (..., 3), angle dimension collapsed

    # Optionally normalize total power to 1:
    if normalize:
        total_power = jnp.sum(intensities_sum)
        intensities_sum = intensities_sum / (total_power + 1e-16)

    return intensities_sum

def universal_compensator_modes(swing: float) -> Array:
    """Compute universal compensator modes and adjust their phases."""
    uc_modes = jnp.array([
        [jnp.pi / 2, jnp.pi],
        [jnp.pi / 2 + swing, jnp.pi],
        [jnp.pi / 2, jnp.pi + swing],
        [jnp.pi / 2, jnp.pi - swing],
        [jnp.pi / 2 - swing, jnp.pi],
    ])
    # Create a dummy field to apply the compensator operator
    field = cf.plane_wave((1, 1), 1, 1, amplitude=cf.linear(0.0))
    field = jax.vmap(cf.universal_compensator, in_axes=(None, 0, 0))(
        field, uc_modes[:, 0], uc_modes[:, 1]
    )
    amplitudes = field.u.squeeze()
    # phase_adjust = jnp.exp(-1j * jnp.angle(amplitudes[:, 1]))
    phase_adjust = jnp.exp(1j * jnp.angle(amplitudes[:, 1]))
    amplitudes = amplitudes * phase_adjust[:, None] * 2.0
    return amplitudes

def create_pupil_field(
    shape,
    dx,
    spectrum,
    positions,
    polarizations
):
    """
    Create a 2D pupil field using VectorField with shape=(B, H, W, C, 3).
    
    Parameters
    ----------
    shape : tuple
        (H, W) size of the pupil plane.
    dx : float or tuple of float
        Physical sampling (ignored here if you just want a placeholder).
    spectrum : float
        Wavelength or frequency (as used by your VectorField).
    positions : array of shape (N, 2)
        The pixel locations where you want to place a source/polarization in the pupil.
    polarizations : array
        The complex polarization vectors to assign at the given positions.
        Can be:
        - A single vector of shape (3,) for all positions: [Ez, Ey, Ex]
        - An array of shape (N, 3) where each row is [Ez, Ey, Ex] for each position

    Returns
    -------
    field : VectorField
        A VectorField with nonzero E only at the specified positions,
        having the requested polarization.
    """

    field = VectorField.create(dx=dx, spectrum=spectrum, spectral_density=1.0, shape=shape)

    # Convert positions to JAX array if it's not already
    positions = jnp.asarray(positions)
    
    # Convert polarizations to JAX array if it's not already
    polarizations = jnp.asarray(polarizations, dtype=jnp.complex64)
    
    # Handle single polarization case
    if polarizations.ndim == 1:
        # Broadcast single polarization to all positions
        polarizations = jnp.tile(polarizations[None, :], (positions.shape[0], 1))
    
    # Ensure we have the right number of polarizations
    if polarizations.shape[0] != positions.shape[0]:
        raise ValueError(f"Number of polarizations ({polarizations.shape[0]}) must match number of positions ({positions.shape[0]})")

    # Update the .u array at each position
    u_data = field.u
    for i in range(positions.shape[0]):
        pos = positions[i]
        pol = polarizations[i]
        
        row, col = pos
        row_int = int(round(row))
        col_int = int(round(col))

        print(f"Setting polarization {pol} at ({row_int}, {col_int})")
        # print norm of polarization
        print(f"Norm of polarization: {jnp.linalg.norm(pol):.2f}")

        # Insert the polarization at (row_int, col_int)
        u_data = u_data.at[0, row_int, col_int, 0, :].set(pol)

        field_intensity = jnp.sum(jnp.abs(u_data[0, row_int, col_int, 0, :])**2)
        print(f"Field intensity at point: {field_intensity:.2f}")

    field = field.replace(u=u_data)

    intensity = field.intensity
    print(f"Intensity maximum of pupil field: {intensity.max():.2f}")

    # Normalize the power of the field
    power = field.power.squeeze()
    if power > 0:
        field = field.replace(u=field.u / jnp.sqrt(power))
        print(f"Normalized field power from {power:.2f} to {field.power.squeeze():.2f}")

    return field

def gaussian_spot_field(
    shape: tuple[int, int],
    dx: float,
    spectrum: float,
    spectral_density: float,
    waist: float,
    center: tuple[float, float] = (0.0, 0.0),
    power: float = 1.0,
    amplitude: float = 1.0,
    polarization: jnp.ndarray | None = None,
    phase_offset: float = 0.0,
) -> VectorField:
    """
    Creates a VectorField representing a spatially localized Gaussian spot.

    The field is created on a grid of given shape and spacing, with a Gaussian envelope
    of waist (in the same units as dx) centered at `center`. A uniform polarization is applied,
    so that the returned field has shape (1, H, W, 1, 3) and is a proper VectorField.

    Args:
        shape: (height, width) of the grid.
        dx: Sampling spacing (assumed isotropic) in physical units.
        spectrum: Wavelength (in µm) of the field.
        spectral_density: Weight for the wavelength.
        waist: Gaussian waist (1/e radius) of the spot.
        center: (x0, y0) coordinates where the Gaussian is centered.
        power: The total power to which the field is normalized.
        amplitude: An overall amplitude scaling.
        polarization: A 3-element array specifying the polarization direction.
                     If None, defaults to x-polarization: [1, 0, 0].
        phase_offset: An overall phase offset.

    Returns:
        A VectorField instance with the Gaussian spot.
    """
    field = VectorField.create(dx, spectrum, spectral_density, shape=shape)

    # Default polarization: x-polarized.
    if polarization is None:
        polarization = jnp.array([1.0, 0.0, 0.0], dtype=field.u.dtype)
    else:
        polarization = jnp.asarray(polarization, dtype=field.u.dtype)

    # Grid coordinates (Y, X).
    Y, X = field.grid[0, 0, ..., 0, 0], field.grid[1, 0, ..., 0, 0]

    # Shift coordinates by the desired center and form the Gaussian envelope.
    X_shifted = X - center[0]
    Y_shifted = Y - center[1]
    envelope = jnp.exp(-((X_shifted**2 + Y_shifted**2) / (waist**2)))

    # Add a phase offset if desired.
    envelope = envelope * jnp.exp(1j * phase_offset)

    # Broadcast to match (1, H, W, 1, 3) shape.
    envelope_expanded = envelope[None, :, :, None]
    u = amplitude * envelope_expanded[..., None] * polarization

    # Update field and normalize to the specified power.
    field = field.replace(u=u)
    # field = field * jnp.sqrt(power / field.power)
    return field

# %% PolScope Class Definition
class PolScope(PyTreeNode):
    # System parameters
    shape: tuple[int, int] = (128, 128)
    spacing: float = 0.546 / 2  # [µm]
    wavelength: float = 0.546  # [µm]
    swing: float = 2 * jnp.pi * 0.03
    num_angles: int = 40

    # Condenser parameters
    condenser_f: float = 5000 
    condenser_n: float = 1.33
    condenser_NA: float = 0.8

    # Objective lens parameters
    objective_f: float = 5000  # [µm]
    objective_n: float = 1.33  # water immersion
    objective_NA: float = 0.8

    # Tube lens parameters
    tube_f: float = 200_000 # gives 20x mag with 5000 µm objective
    tube_n: float = 1.0
    tube_NA: float = 1.0

    # Microlens Array (MLA) parameters
    mla_n: float = 1.0
    mla_f: float = 2500
    mla_radius: float = 50
    mla_separation: float = 100
    mla_n_y: float = 9
    mla_n_x: float = 9

    # Camera parameters
    camera_shape: tuple[int, int] = (144, 144) # 16 * lenslets
    camera_pitch: float = 6.5  # [µm]

    @property
    def angles(self):
        return jnp.atleast_1d(
            fibonacci_cone_sampling(
                num_angles=self.num_angles,
                NA=self.condenser_NA,
                n=self.condenser_n,
                overshoot_factor=10
            )
        )
    
    @property
    def kohler_points(self):
        # Generate the normalized points in a disk of radius NA/n
        points = generate_kohler_2d_points(
            num_angles=self.num_angles,
            NA=self.condenser_NA,
            n=self.condenser_n
        )

        print(f"points:\n{points}")

        # lens_radius = f * tan(arcsin(NA/n))
        lens_radius = self.condenser_f * jnp.tan(jnp.arcsin(self.condenser_NA / self.condenser_n))
        print(f"lens_radius: {lens_radius:.2f} µm")

        # The original points are in a disk of radius (NA/n) in normalized units.
        # To convert them to physical coordinates (in microns) on the pupil plane:
        normalized_max = self.condenser_NA / self.condenser_n
        physical_scaling = lens_radius / normalized_max
        print(f"physical_scaling: {physical_scaling:.2f} µm")
        y_microns = points[:, 0] * physical_scaling
        x_microns = points[:, 1] * physical_scaling
        scaled_points = jnp.column_stack((y_microns, x_microns))
        # print(f"scaled_points before placement:\n{scaled_points}")
        
        # # Map these physical coordinates onto your simulation grid.
        # # Convert physical units (microns) to pixel units using self.spacing.
        # height, width = self.shape
        # y_scaled = y_microns / self.spacing + height // 2
        # x_scaled = x_microns / self.spacing + width // 2
        
        # # Ensure points are within bounds -- maybe omit
        # y_scaled = jnp.clip(y_scaled, 0, height - 1)
        # x_scaled = jnp.clip(x_scaled, 0, width - 1)
        
        # # Stack into a single array of (y, x) coordinates
        # scaled_points = jnp.column_stack((y_scaled, x_scaled))

        # Print original and scaled points side by side
        print("\nOriginal points (normalized)  |  Scaled points (microns)")
        print("-" * 55)
        for orig, scaled in zip(points, scaled_points):
            print(f"({orig[0]:8.3f}, {orig[1]:8.3f})      |  ({scaled[0]:8.3f}, {scaled[1]:8.3f})")
        
        return scaled_points

    def __call__(self, potential: ArrayLike) -> Array:
        uc_modes = universal_compensator_modes(self.swing)
        # print("uc_modes:", uc_modes)
        return jax.vmap(self.forward_chunked, in_axes=(0, None))(uc_modes, potential)

    def forward(self, uc_mode: Array, potential: Array) -> Array:
        """Simulate the full optical propagation."""
        z_sample = potential.shape[0] * self.spacing

        # We'll store the *final* fields for each angle
        list_of_fields = []

        for angle in self.angles:
            field = self.single_angle_forward(uc_mode, potential, angle)
            list_of_fields.append(field)

        field_incoherent = sum_incoherently(list_of_fields)

        # Ey = field_incoherent.u[..., 1]
        # Ex = field_incoherent.u[..., 2]

        # # Calculate intensity from transverse components only:
        # intensity_transverse = jnp.abs(Ex)**2 + jnp.abs(Ey)**2

        camera = init_plane_resample(self.camera_shape, self.camera_pitch)(
            field_incoherent.intensity.squeeze(0),  
            field_incoherent.dx.squeeze()
        )

        return field_incoherent, camera.squeeze()

    def single_angle_forward(self, uc_mode: jnp.ndarray, potential: jnp.ndarray, angle: jnp.ndarray):
        """
        Run the optical propagation for a single angle.

        Args:
            uc_mode:  The universal-compensator mode amplitude (scalar or array).
            potential:  3D potential or something describing the sample.
            angle:   jnp.array([theta, phi]) for this illumination.

        Returns:
            field:  The final Field after microlens array, etc.
        """
        # Determine sample thickness in physical space
        z_sample = potential.shape[0] * self.spacing

        # 1) Create illumination
        field = gaussian_spot_field(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            center=kohler_pt,
            polarization=uc_mode, #(0.0, 1.0j, 1.0)
            spectral_density=1.0,
            waist=1.0 / 4
        )
        field = field.replace(u=field.u)
        pupil_width_um = field.u.shape[2] * field.dx.squeeze()[1]
        # fig = plot_propagated_field(field, title="Pupil")
        field = cf.ff_lens_debye_chunked(field, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um*1.5, num_samples=256, chunk_size=1024)

        # 2) Sample interaction
        field = thick_polarised_sample(field, potential, self.objective_n, self.spacing, NA=self.objective_NA)
        # fig = plot_propagated_field_separate_colorbars(field, title="Sample")

        # 3) Propagate half sample thickness
        field = cf.transfer_propagate(field, -z_sample / 2, self.objective_n, 256, mode="same")
        # fig = plot_propagated_field_separate_colorbars(field, title="Propagate half sample thickness")

        # 4) Analyzer
        M_wave_2x2 = jnp.array([
            [1.0/jnp.sqrt(2),   1j/jnp.sqrt(2)],
            [-1j/jnp.sqrt(2),   1.0/jnp.sqrt(2)]
        ], dtype=jnp.complex64)
        field = apply_jones_in_lab_basis(
            field,
            M_lab_2x2=M_wave_2x2,
        )

        # 5) Microlens
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
            block_between=True,
        )
        # 6) Transfer propagate after MLA
        field = cf.transfer_propagate(field, self.mla_f, self.mla_n, 256, mode="same")

        return field

    def forward_chunked(self, uc_mode: jnp.ndarray, potential: jnp.ndarray, chunk_size: int = 5):
        """
        Instead of vmapping over *all* angles at once, we chunk the angles
        to avoid creating extremely large intermediate arrays.
        """
        def per_angle(a):
            field = self.single_angle_forward(uc_mode, potential, a)
            return field.u

        kohler_points = self.kohler_points
        # We accumulate final_amps in chunks
        angle_batches = []
        for i in range(0, len(kohler_points), chunk_size):
            chunk = kohler_points[i : i + chunk_size]
            chunk_amps = jax.vmap(per_angle, in_axes=0)(chunk)
            angle_batches.append(chunk_amps)

        # Concatenate all chunk results along axis=0 (angle-axis)
        final_amps = jnp.concatenate(angle_batches, axis=0)

        I_incoh = incoherent_sum_across_angles(final_amps, normalize=True)

        # Create a reference field with zero uc_mode
        ref_field = self.single_angle_forward(jnp.zeros_like(uc_mode), potential, kohler_points[0])
        u_incoh = jnp.sqrt(I_incoh) * jnp.exp(1j * 0.0)
        incoherent_field = ref_field.replace(u=u_incoh)

        # Ey = incoherent_field.u[..., 1]
        # Ex = incoherent_field.u[..., 2]

        # # Calculate intensity from transverse components only:
        # intensity_transverse = jnp.abs(Ex)**2 + jnp.abs(Ey)**2

        # camera = init_plane_resample(self.camera_shape, self.camera_pitch)(
        #     intensity_transverse.squeeze(0),
        #     incoherent_field.dx.squeeze()
        # )
        camera = init_plane_resample(self.camera_shape, self.camera_pitch)(
            incoherent_field.intensity.squeeze(0),
            incoherent_field.dx.squeeze(),
        )

        return incoherent_field, camera.squeeze()

    def _store_intermediate(self, 
                            intermediates: dict, 
                            step_name: str, 
                            field_or_array, 
                            record: bool,
                            resample: bool = False):
        """Helper to store an intermediate result if `record` is True."""
        if record:
            if resample:
                # res_factor = 2
                # spacing_record = self.spacing * res_factor
                # shape_record = (self.shape[0] // res_factor, self.shape[1] // res_factor)
                shape_record = (field_or_array.u.shape[1] // 2, field_or_array.u.shape[2] // 2)
                spacing_record = field_or_array.dx.squeeze()[0] * 2
                # shape_record = (128, 128)
                # spacing_record = self.wavelength
                print(f"spacing_record: {spacing_record}")
                print(f"shape_record: {shape_record}")
                resample_fcn = init_plane_resample(out_shape=shape_record, out_spacing=spacing_record)
                field_resampled = resample_fcn(field_or_array.u.squeeze(0), field_or_array.dx.squeeze())
                intermediates[step_name] = field_resampled[None, ...]
            else:
                intermediates[step_name] = field_or_array.u

    def single_angle_lenses(
        self,
        uc_mode: jnp.ndarray,
        kohler_pt: float,
    ):
        """
        Run the optical propagation for a single angle.

        Args:
            uc_mode: Amplitude for the universal-compensator mode (scalar or array).
            kohler_pt: jnp.array([y, x]) for this illumination.

        Returns:
            field: Final Field object after all steps
        """
        field = gaussian_spot_field(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            center=kohler_pt,
            polarization=uc_mode, #(0.0, 1.0j, 1.0)
            spectral_density=1.0,
            waist=1.0 / 6
        )
        pupil_width_um = field.u.shape[2] * field.dx.squeeze()[1]
        fig = plot_propagated_field(field, title="Pupil")

        res = 1024

        field = cf.ff_lens_debye_chunked(field, self.objective_f, self.objective_n, self.objective_NA, range_um=pupil_width_um*2, num_samples=res, chunk_size=1024)
        fig = plot_propagated_field_separate_colorbars(field, title="Condenser lens")

        objective_field = cf.ff_lens_debye_chunked(field, self.objective_f, self.objective_n, self.objective_NA, range_um=pupil_width_um*2, num_samples=res, chunk_size=1024)
        fig = plot_propagated_field_separate_colorbars(objective_field, title="Objective lens")

        tube_field = cf.ff_lens_debye_chunked(objective_field, self.objective_f, self.objective_n, self.objective_NA, range_um=pupil_width_um*2, num_samples=res, chunk_size=1024)
        fig = plot_propagated_field_separate_colorbars(tube_field, title="Tube lens")

        obj_u = objective_field.u
        tube_u = tube_field.u
        field_diff_tube_obj = tube_u - obj_u
        field = field.replace(u=field_diff_tube_obj)
        fig = plot_propagated_field_separate_colorbars(field, title="Field difference between tube and objective")

        intensity_diff = objective_field.intensity - tube_field.intensity
        fig = plt.figure()
        plt.title("Intensity difference between tube and objective lenses")
        plt.imshow(intensity_diff.squeeze())
        plt.colorbar()
        plt.show()

        return field

    def single_angle_illum_analyzer(
        self,
        uc_mode: jnp.ndarray,
        kohler_pt: float,
    ):
        """
        Run the optical propagation for a single angle.

        Args:
            uc_mode: Amplitude for the universal-compensator mode (scalar or array).
            kohler_pt: jnp.array([y, x]) for this illumination.

        Returns:
            field: Final Field object after all steps
        """
        M_wave_2x2_linear = jnp.array([
            [1.0, 0.0],
            [0.0, 0.0]
        ], dtype=jnp.complex64)
        field = gaussian_spot_field(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            center=kohler_pt,
            polarization=uc_mode, #(0.0, 1.0j, 1.0)
            spectral_density=1.0,
            waist=1.0 / 6
        )
        pupil_width_um = field.u.shape[2] * field.dx.squeeze()[1]
        fig = plot_propagated_field(field, title="Pupil")

        M_wave_2x2 = jnp.array([
            [1.0/jnp.sqrt(2),   1j/jnp.sqrt(2)],
            [-1j/jnp.sqrt(2),   1.0/jnp.sqrt(2)]
        ], dtype=jnp.complex64)
        field_analyzer_after_pupil = apply_jones_in_lab_basis(field, M_lab_2x2=M_wave_2x2_linear)
        fig = plot_propagated_field(field_analyzer_after_pupil, title="Analyzer after pupil")

        res = 256

        field = cf.ff_lens_debye_chunked(field, self.objective_f, self.objective_n, self.objective_NA, range_um=pupil_width_um, num_samples=res, chunk_size=1024)
        fig = plot_propagated_field_separate_colorbars(field, title="Condenser lens")

        M_wave_2x2 = jnp.array([
            [1.0/jnp.sqrt(2),   -1j/jnp.sqrt(2)],
            [1j/jnp.sqrt(2),   1.0/jnp.sqrt(2)]
        ], dtype=jnp.complex64)
        # field_analyzer_after_condenser = apply_jones_in_lab_basis(field, M_lab_2x2=M_wave_2x2)
        angle = compute_angles_from_offset(kohler_pt[1], kohler_pt[0], self.objective_f)
        print(f"Angle: {angle}")
        field_analyzer_after_condenser = apply_jones_in_wave_basis(
            field,
            M_wave_2x2_linear,
            n_medium=self.objective_n,
            spectrum=self.wavelength,
            angle=angle
        )
        fig = plot_propagated_field_separate_colorbars(field_analyzer_after_condenser, title="Analyzer after condenser (wave basis)")
        field_analyzer_after_condenser = apply_jones_in_lab_basis(
            field_analyzer_after_condenser,
            M_lab_2x2=M_wave_2x2_linear,
        )
        fig = plot_propagated_field_separate_colorbars(field_analyzer_after_condenser, title="Analyzer after condenser (lab basis)")

        if True:
            field = cf.ff_lens_debye_chunked(field, self.objective_f, self.objective_n, self.objective_NA, range_um=pupil_width_um, num_samples=res, chunk_size=1024)
            fig = plot_propagated_field_separate_colorbars(field, title="Objective lens")

            field = cf.ff_lens_debye_chunked(field, self.objective_f, self.objective_n, self.objective_NA, range_um=pupil_width_um, num_samples=res, chunk_size=1024)
            fig = plot_propagated_field_separate_colorbars(field, title="Tube lens")

            M_wave_2x2 = jnp.array([
                [1.0/jnp.sqrt(2),   1j/jnp.sqrt(2)],
                [-1j/jnp.sqrt(2),   1.0/jnp.sqrt(2)]
            ], dtype=jnp.complex64)
            field = apply_jones_in_lab_basis(field, M_lab_2x2=M_wave_2x2_linear)
            fig = plot_propagated_field_separate_colorbars(field, title="Analyzer")

        return field

    def single_angle_forward_intermediates(
        self,
        uc_mode: jnp.ndarray,
        potential: jnp.ndarray,
        kohler_pt: float,
        return_intermediates: bool = False
    ):
        """
        Run the optical propagation for a single angle, optionally returning intermediate fields.

        Args:
            uc_mode: Amplitude for the universal-compensator mode (scalar or array).
            potential: 3D potential describing the sample.
            kohler_pt: jnp.array([y, x]) for this illumination.
            return_intermediates: Whether to collect and return fields at each step.

        Returns:
            field: Final Field object after all steps.
            intermediates: dict of {step_name -> Field}, if return_intermediates=True
        """
        z_sample = potential.shape[0] * self.spacing
        intermediates = {}

        # Helper for recording steps
        def record(step_name: str, fld):
            self._store_intermediate(intermediates, step_name, fld, return_intermediates, resample=False) # adjusting fld.u to be fld

        # 1) Create illumination
        field = gaussian_spot_field(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            center=kohler_pt,
            polarization=uc_mode, #(0.0, 1.0j, 1.0)
            spectral_density=1.0,
            waist=1.0 / 4
        )
        pupil_width_um = field.u.shape[2] * field.dx.squeeze()[1]
        record("pupil", field)
    
        # num_samples must match that of the sample
        sample_pixels = potential.shape[1]
        field = cf.ff_lens_debye_chunked(field, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um*1.5, num_samples=sample_pixels, chunk_size=1024)
        record("condenser_lens", field)

        if False:
            # 2) Sample interaction
            field = thick_polarised_sample(field, potential, self.objective_n, self.spacing, NA=self.objective_NA)
            record("sample_interaction", field)

            # 3) Propagate half sample thickness
            field = cf.transfer_propagate(field, -z_sample / 2, self.objective_n, 256, mode="same")
            record("propagate_half", field)

        # 5) Analyzer
        M_wave_2x2 = jnp.array([
            [1.0/jnp.sqrt(2),   1j/jnp.sqrt(2)],
            [-1j/jnp.sqrt(2),   1.0/jnp.sqrt(2)]
        ], dtype=jnp.complex64)
        field = apply_jones_in_lab_basis(
            field,
            M_lab_2x2=M_wave_2x2,
        )
        record("analyzer", field)

        # 7) Microlens array
        # print(f"field.dx.squeeze()[0]: {field.dx.squeeze()[0]}")
        time_start = time.time()
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
            block_between=True,
        )
        record("mla", field)

        # 8) Transfer propagate after MLA
        field = cf.transfer_propagate(field, self.mla_f, self.mla_n, 512, mode="same")
        record("final_propagate", field)

        if return_intermediates:
            return field, intermediates
        else:
            return field

    def single_angle_forward_intermediates_all_steps(
        self,
        uc_mode: jnp.ndarray,
        potential: jnp.ndarray,
        kohler_pt: float,
        return_intermediates: bool = False
    ):
        """
        Run the optical propagation for a single angle, optionally returning intermediate fields.

        Args:
            uc_mode: Amplitude for the universal-compensator mode (scalar or array).
            potential: 3D potential describing the sample.
            kohler_pt: jnp.array([y, x]) for this illumination.
            return_intermediates: Whether to collect and return fields at each step.

        Returns:
            field: Final Field object after all steps.
            intermediates: dict of {step_name -> Field}, if return_intermediates=True
        """
        z_sample = potential.shape[0] * self.spacing
        intermediates = {}

        # Helper for recording steps
        def record(step_name: str, fld):
            self._store_intermediate(intermediates, step_name, fld, return_intermediates, resample=False) # adjusting fld.u to be fld

        # # 1) Create illumination
        # field = single_angle_illumination(
        #     shape=self.shape,
        #     dx=self.spacing,
        #     spectrum=self.wavelength,
        #     n_medium=self.condenser_n,
        #     angle=angle,
        #     amplitude=uc_mode,
        # )
        # record("illumination", field)
        # field = field.replace(u=field.u.astype(jnp.complex128))
        # kohler_pt = jnp.array([kohler_pt])
        # field = create_pupil_field(
        #     shape=self.shape,
        #     dx=self.spacing,
        #     spectrum=self.wavelength,
        #     positions=kohler_pt,
        #     polarizations=uc_mode
        # )
        field = gaussian_spot_field(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            center=kohler_pt,
            polarization=uc_mode, #(0.0, 1.0j, 1.0)
            spectral_density=1.0,
            waist=1.0 / 4
        )
        pupil_width_um = field.u.shape[2] * field.dx.squeeze()[1]
        # fig = plot_propagated_field(field, title="Pupil")
        record("pupil", field)
    
        # num_samples must match that of the sample
        field = cf.ff_lens_debye_chunked(field, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um*1.5, num_samples=512, chunk_size=1024)
        # fig = plot_propagated_field_separate_colorbars(field, title="Condenser lens")
        record("condenser_lens", field)

        if False:
            # 2) Sample interaction
            field = thick_polarised_sample(field, potential, self.objective_n, self.spacing, NA=self.objective_NA)
            # fig = plot_propagated_field_separate_colorbars(field, title="Sample")
            record("sample_interaction", field)

            # 3) Propagate half sample thickness
            field = cf.transfer_propagate(field, -z_sample / 2, self.objective_n, 256, mode="same")
            # fig = plot_propagated_field_separate_colorbars(field, title="Propagate half sample thickness")
            record("propagate_half", field)

        if False:
            # 4) Objective lens
            # field = cf.ff_lens_debye_chunked(field, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um*1.5, num_samples=512, chunk_size=1024)

            field = cf.ff_lens_debye_chunked(field, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um*0.5, num_samples=512, chunk_size=1024)
            record("objective_lens", field)
            # fig = plot_propagated_field_separate_colorbars(field, title="Objective lens")
            # plt.savefig(f"{SAVE_DIR}/objective_lens_{angle[0]}.png")

        if True:
            # 5) Analyzer
            M_wave_2x2 = jnp.array([
                [1.0/jnp.sqrt(2),   1j/jnp.sqrt(2)],
                [-1j/jnp.sqrt(2),   1.0/jnp.sqrt(2)]
            ], dtype=jnp.complex64)
            # Switch sign of analyzer if applying in wave basis
            # field = apply_jones_in_wave_basis(
            #     field,
            #     M_wave_2x2,
            #     n_medium=self.condenser_n,
            #     spectrum=self.wavelength,
            #     angle=angle
            # )
            field = apply_jones_in_lab_basis(
                field,
                M_lab_2x2=M_wave_2x2,
            )
            # fig = plot_propagated_field_separate_colorbars(field, title="Analyzer")
            record("analyzer", field)

        if False:
            # 6) Tube lens
            # field = cf.ff_lens_debye_chunked(field, self.objective_f, self.objective_n, self.objective_NA, range_um=pupil_width_um*1.5, num_samples=256, chunk_size=1024)
            field = cf.ff_lens_debye_chunked(field, self.objective_f, self.objective_n, self.objective_NA, range_um=pupil_width_um*1.5, num_samples=256, chunk_size=1024)
            # field = cf.ff_lens(field, self.tube_f, self.tube_n, self.tube_NA)
            record("tube_lens (repeat of objective)", field)
            fig = plot_propagated_field_separate_colorbars(field, title="Tube lens")
            # field = cf.ff_lens_debye_chunked(field, self.tube_f, self.tube_n, self.tube_NA, range_um=pupil_width_um*1.5, num_samples=128, chunk_size=1024)
        
            # record("tube_lens", field)

        if True:
            # 7) Microlens array
            # print(f"field.dx.squeeze()[0]: {field.dx.squeeze()[0]}")
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
                block_between=True,
            )
            record("mla", field)
            # fig = plot_propagated_field_separate_colorbars(field, title="Microlens array")
            # 8) Transfer propagate after MLA
            field = cf.transfer_propagate(field, self.mla_f, self.mla_n, 512, mode="same")
            record("final_propagate", field)
            # fig = plot_propagated_field_separate_colorbars(field, title="Final propagate")
        if return_intermediates:
            return field, intermediates
        else:
            return field

    def single_angle_forward_plot(
        self,
        uc_mode: jnp.ndarray,
        potential: jnp.ndarray,
        kohler_pt: float,
        return_intermediates: bool = False
    ):
        """
        Run the optical propagation for a single angle, optionally returning intermediate fields.

        Args:
            uc_mode: Amplitude for the universal-compensator mode (scalar or array).
            potential: 3D potential describing the sample.
            kohler_pt: jnp.array([y, x]) for this illumination.
            return_intermediates: Whether to collect and return fields at each step.

        Returns:
            field: Final Field object after all steps.
            intermediates: dict of {step_name -> Field}, if return_intermediates=True
        """

        field = gaussian_spot_field(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            center=kohler_pt,
            polarization=uc_mode, #(0.0, 1.0j, 1.0)
            spectral_density=1.0,
            waist=1.0 / 4
        )
        pupil_width_um = field.u.shape[2] * field.dx.squeeze()[1]
        fig = plot_propagated_field(field, title="Pupil")

        num_samples_condenser = 1024
        field_size_condenser = pupil_width_um * 2
        field = cf.ff_lens_debye_chunked(field, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=field_size_condenser, num_samples=num_samples_condenser, chunk_size=1024)
        fig = plot_propagated_field_separate_colorbars(field, title="Condenser lens")
        print(f"After condenser lens, field.dx: {field.dx.squeeze()[0]:.3f} um")

        if False:
            # 2) Sample interaction
            dz = field.dx.squeeze()[0] #self.spacing
            # dz = self.spacing
            field = thick_polarised_sample(field, potential, self.objective_n, dz, NA=self.objective_NA)
            fig = plot_propagated_field_separate_colorbars(field, title="Sample")
            # 3) Propagate half sample thickness
            z_sample = potential.shape[0] * dz
            field = cf.transfer_propagate(field, -z_sample / 2, self.objective_n, 256, mode="same")
            fig = plot_propagated_field_separate_colorbars(field, title="Propagate half sample thickness")

        if False:
            # 4) Objective lens
            # field = cf.ff_lens_debye_chunked(field, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um*1.5, num_samples=512, chunk_size=1024)
            field = cf.ff_lens_debye_chunked(field, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um*0.5, num_samples=512, chunk_size=1024)
            fig = plot_propagated_field_separate_colorbars(field, title="Objective lens")
            # plt.savefig(f"{SAVE_DIR}/objective_lens_{angle[0]}.png")

        if True:
            # 5) Analyzer
            M_wave_2x2 = jnp.array([
                [1.0/jnp.sqrt(2),   1j/jnp.sqrt(2)],
                [-1j/jnp.sqrt(2),   1.0/jnp.sqrt(2)]
            ], dtype=jnp.complex64)
            field = apply_jones_in_lab_basis(
                field,
                M_lab_2x2=M_wave_2x2,
            )
            fig = plot_propagated_field_separate_colorbars(field, title="Analyzer")

        if False:
            # 6) Tube lens
            field = cf.ff_lens_debye_chunked(field, self.objective_f, self.objective_n, self.objective_NA, range_um=pupil_width_um*1.5, num_samples=256, chunk_size=1024)
            record("tube_lens (repeat of objective)", field)
            fig = plot_propagated_field_separate_colorbars(field, title="Tube lens")
        

        if True:
            # 7) Microlens array
            mla_radius = jnp.floor(self.mla_radius / field.dx.squeeze()[0]) * field.dx.squeeze()[0]
            mla_separation = jnp.floor(self.mla_separation / field.dx.squeeze()[0]) * field.dx.squeeze()[0]
            print(f"mla_radius: {mla_radius:.3f} um, mla_separation: {mla_separation:.3f} um, field.dx: {field.dx.squeeze()[0]:.3f} um")
            field = cf.rectangular_microlens_array(
                field,
                n=self.mla_n,
                f=self.mla_f,
                num_lenses_height=self.mla_n_y,
                num_lenses_width=self.mla_n_x,
                radius=mla_radius,
                separation=mla_separation,
                block_between=True,
            )
            fig = plot_propagated_field_separate_colorbars(field, title="Microlens array")
            camera = init_plane_resample(self.camera_shape, self.camera_pitch)(
                field.intensity.squeeze(0),
                field.dx.squeeze()
            )

            max_plot = camera.max() #/ 2
            plt.imshow(camera.squeeze(), vmin=0, vmax=max_plot, cmap='inferno')
            plt.colorbar()
            plt.title(f"Camera Field (before propagate)")
            plt.show()
            # 8) Transfer propagate after MLA
            field = cf.transfer_propagate(field, self.mla_f, self.mla_n, 512, mode="same")
            print(f"After final propagate, field.dx.squeeze()[0]: {field.dx.squeeze()[0]:.3f} um")
            fig = plot_propagated_field_separate_colorbars(field, title=f"Final propagate")

            camera = init_plane_resample(self.camera_shape, self.camera_pitch)(
                field.intensity.squeeze(0),
                field.dx.squeeze()
            )

            max_plot = camera.max() #/ 2
            plt.imshow(camera.squeeze(), vmin=0, vmax=max_plot, cmap='inferno')
            plt.colorbar()
            plt.title(f"Camera Field")
            plt.show()
            # fig.savefig(f"{SAVE_DIR}/mode0_sample_camera.png", bbox_inches='tight')

        return field

    def forward_debug(self, uc_mode: Array, potential: Array):
        """
        Forward pass that collects the intermediate fields for each angle,
        sums them incoherently, and plots them at each step.
        """
        reference_field = VectorField.create(self.spacing, self.wavelength, 1, shape=self.shape)
        all_angles_intermediates = []
        for kohler_pt in tqdm(self.kohler_points, desc="Processing angles", disable=jax.config.jax_disable_jit):
            final_field, intermediates = self.single_angle_forward_intermediates(
                uc_mode, potential, kohler_pt, return_intermediates=True
            )
            all_angles_intermediates.append(intermediates)

        step_names = list(all_angles_intermediates[0].keys())

        # For each step, gather fields from all angles and do an incoherent sum
        for step_name in step_names:
            step_fields_u = [inter[step_name] for inter in all_angles_intermediates]
            step_incoh_u = incoherent_sum_across_angles(
                step_fields_u, normalize=False
            )
            step_incoh_field = reference_field.replace(u=step_incoh_u)
            # fig = plot_propagated_field_separate_colorbars(step_incoh_field, title=f"Incoherent sum at step {step_name}", presquared=True)
            fig = plot_propagated_field(step_incoh_field, title=f"Incoherent sum at step {step_name}", presquared=True)
            plt.tight_layout()
            fig.savefig(f"{SAVE_DIR}/mode0_sample_{step_name}.png", bbox_inches='tight')

        final_fields_u = [inter["final_propagate"] for inter in all_angles_intermediates]
        final_sum_u = incoherent_sum_across_angles(final_fields_u, normalize=False)
        final_sum = reference_field.replace(u=final_sum_u)
        camera = init_plane_resample(self.camera_shape, self.camera_pitch)(
            final_sum.intensity.squeeze(0),
            final_sum.dx.squeeze()
        )
        # plot_propagated_field(final_sum, title="Final Field (Incoherent Sum)") # Repeated

        intensity = final_sum.intensity
        max_plot = intensity.max() / 100
        plt.imshow(intensity.squeeze(), vmin=0, vmax=max_plot, cmap='inferno')
        plt.colorbar()
        plt.title("Final Field (after adjusting for contrast)")
        plt.show()

        print(f"Camera field max: {camera.max()}, camera shape: {camera.shape}")
        max_plot = camera.max() #/ 2
        plt.imshow(camera.squeeze(), vmin=0, vmax=max_plot, cmap='inferno')
        plt.colorbar()
        plt.title("Camera Field")
        plt.show()
        fig.savefig(f"{SAVE_DIR}/mode0_sample_camera.png", bbox_inches='tight')

        return final_sum, camera

    def single_angle_forward_illum_analyzer_debug(
        self,
        uc_mode: jnp.ndarray,
        angle: jnp.ndarray,
        return_intermediates: bool = False
    ):
        """
        Minimal forward simulation for a single angle:
        1) Create illumination (with given uc_mode).
        2) Apply analyzer in local basis.
        """
        intermediates = {}
        
        # Helper to record a step's amplitude field (u) if debugging
        def record(step_name: str, field, store: bool):
            if store:
                intermediates[step_name] = field.u

        # Illumination
        field = single_angle_illumination(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            n_medium=self.condenser_n,
            angle=angle,
            amplitude=(0.0, 1.0j, 1.0)
        )
        record("illumination", field, return_intermediates)

        # Analyzer
        M_wave_2x2 = jnp.array([
            [1.0/jnp.sqrt(2),   1j/jnp.sqrt(2)],
            [-1j/jnp.sqrt(2),   1.0/jnp.sqrt(2)]
        ], dtype=jnp.complex64)
        field = apply_jones_in_wave_basis(
            field,
            M_wave_2x2,
            n_medium=self.condenser_n,
            spectrum=self.wavelength,
            angle=angle
        )
        record("analyzer", field, return_intermediates)

        if return_intermediates:
            return field, intermediates
        else:
            return field

    def forward_debug_minimal(self, uc_mode: jnp.ndarray):
        """
        Forward pass that ONLY does illumination -> analyzer,
        then sums incoherently across angles and plots intermediate results.
        """
        # 1) Make a reference field so we have shape/dx info.
        #    We'll pass zero amplitude & the first angle to get the correct shape.
        reference_field = self.single_angle_forward_illum_analyzer_debug(
            uc_mode=jnp.zeros_like(uc_mode),
            angle=self.angles[0],
            return_intermediates=False
        )

        # 2) Collect intermediate amplitudes for each real angle
        all_angles_intermediates = []
        for angle in self.angles:
            final_field, intermediates = self.single_angle_forward_illum_analyzer_debug(
                uc_mode,
                angle,
                return_intermediates=True
            )
            all_angles_intermediates.append(intermediates)

        # 3) Determine the pipeline steps from the first dictionary
        step_names = list(all_angles_intermediates[0].keys())

        # 4) For each step, gather amplitude arrays from all angles and do an incoherent sum
        for step_name in step_names:
            # A list of arrays, one for each angle
            step_fields_u = [inter[step_name] for inter in all_angles_intermediates]

            # Incoherent sum
            step_incoh_u = incoherent_sum_across_angles(step_fields_u, normalize=False)

            # Build a new field for plotting with the reference's metadata
            step_incoh_field = reference_field.replace(u=step_incoh_u)

            # Plot the result
            plot_propagated_field(
                step_incoh_field,
                title=f"Incoherent sum at step: {step_name}"
            )

        # 5) Also do an incoherent sum of the final fields
        final_fields_u = [inter["analyzer"] for inter in all_angles_intermediates]
        final_sum_u = incoherent_sum_across_angles(final_fields_u, normalize=False)
        final_sum = reference_field.replace(u=final_sum_u)

        # 6) Optionally plot the final field
        plot_propagated_field(final_sum, title="Final (Illum -> Analyzer) Incoherent Sum")

        return final_sum

    def single_angle_forward_cancel_test(self):
        """
        Test if LCP -> RCP analyzer kills the signal for normal incidence.
        """
        # Force angle=0 (normal incidence)
        angle = jnp.array([0.1, 0.5])

        # Force an LCP universal-compensator mode in x-y plane
        # E_x = 1/sqrt(2), E_y = i/sqrt(2), E_z=0
        lcp_mode = jnp.array([
            0.0 + 0.0j,              # Ez
            (1j / jnp.sqrt(2)),      # Ey = i/sqrt(2)
            (1.0 / jnp.sqrt(2))      # Ex = 1/sqrt(2)
        ], dtype=jnp.complex64)

        field = single_angle_illumination(
            shape=(1, *self.shape),
            dx=self.spacing,
            spectrum=self.wavelength,
            n_medium=self.condenser_n,
            angle=angle,
            amplitude=lcp_mode
        )
        print("Illumination field:")
        fig = plot_propagated_field(field)
        # analyzer
        M_wave_2x2 = jnp.array([
            [1.0/jnp.sqrt(2),   -1j/jnp.sqrt(2)],
            [1j/jnp.sqrt(2),   1.0/jnp.sqrt(2)]
        ], dtype=jnp.complex64)
        field = apply_jones_in_wave_basis(
            field,
            M_wave_2x2,
            n_medium=self.condenser_n,
            spectrum=self.wavelength,
            angle=angle
        )
        print("Analyzed field:")
        fig = plot_propagated_field(field)

        amp_norm = jnp.abs(field.u).max()
        print("Final amplitude norm (should be near 0):", amp_norm)

        return field

    def single_angle_forward_illum_debug(
        self,
        uc_mode: jnp.ndarray,
        kohler_pt: jnp.ndarray,
        return_intermediates: bool = False
    ):
        """
        Minimal forward simulation for a single angle that ONLY performs:
        1) Create illumination (with given uc_mode).
        """
        intermediates = {}

        def record(step_name: str, field, store: bool):
            if store:
                intermediates[step_name] = field.u

        print(f"kohler_pt: {kohler_pt}")
        field_pupil = gaussian_spot_field(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            center=kohler_pt,
            polarization=uc_mode, #(0.0, 1.0j, 1.0)
            spectral_density=1.0,
            waist=1.0 / 4
        )
        pupil_width_um = field_pupil.u.shape[2] * field_pupil.dx.squeeze()[1]
        print(f"Pupil field is of width {pupil_width_um:.2f} um")

        record("pupil", field_pupil, return_intermediates)

        fig = plot_propagated_field(field_pupil, title="Pupil Field")

        time_start = time.time()
        field_debye_chunked = cf.ff_lens_debye_chunked(field_pupil, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um*1.5, num_samples=1024, chunk_size=1024)
        record("ff_lens_debye_chunked", field_debye_chunked, return_intermediates)
        time_end = time.time()
        print(f"Time taken for ff_lens_debye_chunked: {time_end - time_start:.2f} seconds")


        fig = plot_propagated_field_separate_colorbars(field_debye_chunked, title="Debye Field")
        fig = plot_propagated_field(field_debye_chunked, title="Debye Field")
    
        if False:
            print(f"Debye field (direct) components:")
            Ezyx = field_debye.u.squeeze()
            print(f"Ez max: {jnp.max(Ezyx[..., 0]):.2f}, min: {jnp.min(Ezyx[..., 0]):.2f}")
            print(f"Ey max: {jnp.max(Ezyx[..., 1]):.2f}, min: {jnp.min(Ezyx[..., 1]):.2f}")
            print(f"Ex max: {jnp.max(Ezyx[..., 2]):.2f}, min: {jnp.min(Ezyx[..., 2]):.2f}")

        time_start = time.time()
        field_debye_2_chunked = cf.ff_lens_debye_chunked(field_debye_chunked, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um, num_samples=128, chunk_size=1024)
        record("second_ff_lens_debye_chunked", field_debye_2_chunked, return_intermediates)
        time_end = time.time()
        print(f"Time taken for second_ff_lens_debye_chunked: {time_end - time_start:.2f} seconds")

        
        fig = plot_propagated_field_separate_colorbars(field_debye_2_chunked, title="Debye 2 Field")
        fig = plot_propagated_field(field_debye_2_chunked, title="Debye 2 Field")
        field = field_debye_2_chunked

        if return_intermediates:
            return field, intermediates
        else:
            return field

    def single_angle_forward_illum_downsampled(
        self,
        uc_mode: jnp.ndarray,
        kohler_pt: jnp.ndarray,
        return_intermediates: bool = False
    ):
        """
        Minimal forward simulation for a single angle that ONLY performs:
        1) Create illumination (with given uc_mode).
        """
        intermediates = {}

        res_factor = 6
        spacing_record = self.spacing * res_factor
        shape_record = (self.shape[0] // res_factor, self.shape[1] // res_factor)
        print(f"spacing_record: {spacing_record}")
        print(f"shape_record: {shape_record}")
        resample_fcn = init_plane_resample(out_shape=shape_record, out_spacing=spacing_record)
        def record(step_name: str, field, store: bool, downsample: bool = True):
            if store:
                # Downsample field.u if required.
                if downsample:
                    # Example: take every nth sample (this is a simple approach)
                    # downsampled_field = field.u[..., ::4, ::4]  # adjust slicing as needed
                    def gaussian_kernel(kernel_size: int, sigma: float) -> jnp.ndarray:
                        # Create a 2D Gaussian kernel.
                        x = jnp.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=jnp.float32)
                        y = x.reshape(-1, 1)
                        kernel = jnp.exp(-(x**2 + y**2) / (2 * sigma**2))
                        kernel = kernel / jnp.sum(kernel)
                        return kernel

                    def apply_lowpass_filter(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
                        batch, H, W, C, D = image.shape
                        print("apply_lowpass_filter:")
                        print("  image.shape =", image.shape)

                        image_reshaped = image.reshape(batch, H, W, C * D)
                        print("  image_reshaped.shape =", image_reshaped.shape)
                        print("  in_channels =", image_reshaped.shape[-1])

                        # Expand kernel to (5,5,1,1), not (5,5,3,3)
                        kernel = kernel.astype(image_reshaped.dtype)
                        kernel_expanded = kernel[:, :, None, None]
                        kernel_expanded = jnp.tile(kernel_expanded, (1, 1, 1, 3))  # now shape (5,5,1,3)
                        print("  kernel_expanded.shape =", kernel_expanded.shape)

                        filtered = lax.conv_general_dilated(
                            image_reshaped,
                            kernel_expanded,
                            window_strides=(1, 1),
                            padding='SAME',
                            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                            feature_group_count=3,
                        )
                        print("  filtered.shape =", filtered.shape)

                        # Reshape back
                        filtered = filtered.reshape(batch, H, W, C, D)
                        return filtered

                    if False:
                        # Example usage:
                        kernel_size = 5  # Adjust based on your requirements.
                        sigma = 1.0      # Standard deviation for Gaussian blur.
                        kernel = gaussian_kernel(kernel_size, sigma)

                        # Assuming field.u is your high-resolution field with shape (1, H, W, 1, 3)
                        filtered_field = apply_lowpass_filter(field.u, kernel)
                        field = field.replace(u=filtered_field)
                    print(step_name)
                    print(f"field.u.shape: {field.u.shape}, field.dx: {field.dx.squeeze()}")
                    downsampled_field = resample_fcn(field.u.squeeze(0), field.dx.squeeze())
                    downsampled_field = downsampled_field[None, ...]
                    print(f"downsampled_field.shape: {downsampled_field.shape}")
                else:
                    downsampled_field = field.u
                intermediates[step_name] = downsampled_field

        print(f"kohler_pt: {kohler_pt}")
        field_pupil = gaussian_spot_field(
            shape=self.shape,
            dx=self.spacing,
            spectrum=self.wavelength,
            center=kohler_pt,
            polarization=uc_mode, #(0.0, 1.0j, 1.0)
            spectral_density=1.0,
            waist=1.0 / 4
        )
        pupil_width_um = field_pupil.u.shape[2] * field_pupil.dx.squeeze()[1]
        print(f"Pupil field is of width {pupil_width_um:.2f} um")

        record("pupil", field_pupil, return_intermediates)

        fig = plot_propagated_field(field_pupil, title="Pupil Field")

        time_start = time.time()
        field_debye = cf.ff_lens_debye_chunked(field_pupil, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um*1.5, num_samples=512, chunk_size=1024)
        record("ff_lens_debye", field_debye, return_intermediates, downsample=True)
        time_end = time.time()
        print(f"Time taken for ff_lens_debye: {time_end - time_start:.2f} seconds")


        fig = plot_propagated_field_separate_colorbars(field_debye, title="Debye Field")
        fig = plot_propagated_field(field_debye, title="Debye Field")
    
        if False:
            print(f"Debye field (direct) components:")
            Ezyx = field_debye.u.squeeze()
            print(f"Ez max: {jnp.max(Ezyx[..., 0]):.2f}, min: {jnp.min(Ezyx[..., 0]):.2f}")
            print(f"Ey max: {jnp.max(Ezyx[..., 1]):.2f}, min: {jnp.min(Ezyx[..., 1]):.2f}")
            print(f"Ex max: {jnp.max(Ezyx[..., 2]):.2f}, min: {jnp.min(Ezyx[..., 2]):.2f}")

        time_start = time.time()
        field_debye_2 = cf.ff_lens_debye_chunked(field_debye, self.condenser_f, self.condenser_n, self.condenser_NA, range_um=pupil_width_um, num_samples=128, chunk_size=1024)
        record("second_ff_lens_debye", field_debye_2, return_intermediates)
        time_end = time.time()
        print(f"Time taken for second_ff_lens_debye: {time_end - time_start:.2f} seconds")

        
        fig = plot_propagated_field_separate_colorbars(field_debye_2, title="Debye 2 Field")
        fig = plot_propagated_field(field_debye_2, title="Debye 2 Field")
        field = field_debye_2

        if return_intermediates:
            return field, intermediates
        else:
            return field


    def forward_debug_minimal_illum_only(self, uc_mode: jnp.ndarray):
        """
        Forward pass that ONLY does illumination across all angles,
        then sums incoherently and plots intermediate results.
        """
        reference_field = VectorField.create(self.spacing, self.wavelength, 1, shape=self.shape)

        # 2) Collect intermediate amplitudes for each real angle
        all_angles_intermediates = []
        for kohler_pt in self.kohler_points:
            final_field, intermediates = self.single_angle_forward_illum_downsampled(
                uc_mode,
                kohler_pt,
                return_intermediates=True
            )
            all_angles_intermediates.append(intermediates)

        # 3) Determine the pipeline steps from the first dictionary (should only be "illumination")
        step_names = list(all_angles_intermediates[0].keys())

        # 4) For each step, gather amplitude arrays from all angles and do an incoherent sum
        for step_name in step_names:
            print(f"Step name: {step_name}")
            step_fields_u = [inter[step_name] for inter in all_angles_intermediates]
            step_incoh_u = incoherent_sum_across_angles(step_fields_u, normalize=False)
            # step_incoh_u = step_fields_u[0]
            print(f"step_incoh_u.shape: {step_incoh_u.shape}")
            step_incoh_field = reference_field.replace(u=step_incoh_u)
            plot_propagated_field(
                step_incoh_field,
                title=f"Incoherent sum at step: {step_name}",
                presquared=True,
                # vmin=0
            )

        return step_incoh_field



# %% Basic output
scope = PolScope()
uc_mode = universal_compensator_modes(scope.swing)[0]
print(f"uc_mode: {uc_mode}")
# field_illum = scope.forward_debug_minimal_illum_only(uc_mode)
wavelength = scope.wavelength
focal_length = 10
NA = scope.condenser_NA
n = scope.condenser_n
diam_um = 2 * focal_length * jnp.tan(jnp.arcsin(NA / n))
res_factor = 6
shape_len = 128 * res_factor
spacing = wavelength / 2 / res_factor
pupil_field_width_um = shape_len * spacing
assert pupil_field_width_um >= diam_um, f"pupil_field_width_um ({pupil_field_width_um:.2f} um) must be >= diam_um ({diam_um:.2f} um)"
scope_short = replace(
    scope,
    shape=(shape_len, shape_len),
    condenser_f=focal_length,
    objective_f=focal_length,
    spacing=spacing,         
)
# field_illum_short = scope_short.single_angle_forward_illum_debug(uc_mode, jnp.array([-7.5, 0]))
field_illum_short = scope_short.forward_debug_minimal_illum_only(uc_mode)
# field_illum_short = scope_short.single_angle_forward_illum_debug(uc_mode, jnp.array([50, 32]))
# field_illum_short = scope_short.single_angle_forward_illum_debug(uc_mode, jnp.array([180, 128]))
 # %%
scope = PolScope()
print(f"scope.kohler_points:\n{scope.kohler_points}")

print("---")
wavelength = scope.wavelength
focal_length = 10
NA = scope.condenser_NA
n = scope.condenser_n
diam_um = 2 * focal_length * jnp.tan(jnp.arcsin(NA / n))
shape_len = 64
spacing = wavelength / 2
pupil_field_width_um = shape_len * spacing
assert pupil_field_width_um >= diam_um, f"pupil_field_width_um ({pupil_field_width_um:.2f} um) must be >= diam_um ({diam_um:.2f} um)"
scope_short = replace(
    scope,
    shape=(shape_len, shape_len),
    condenser_f=focal_length,
    objective_f=focal_length,          
    spacing=spacing,         
)
print(f"scope_short.kohler_points:\n{scope_short.kohler_points}")

# %%
field_illum_short = scope_short.forward_debug_minimal_illum_only(uc_mode)
# %%
field_illum_short = scope_short.forward_debug(uc_mode)

# %%
potential_bg = single_bead_sample(
    n_background=1.33,
    n_bead=jnp.array([1.33, 1.33, 1.33]),
    orientation=jnp.array([jnp.pi / 4, 0, 0]),
    radius=4.0,
    shape=(0, 256, 256),
    spacing=0.546 / 2, 
    k0=2 * jnp.pi / 0.546,
)[:-1, :-1, :-1, None]

#%% Various configurations with a small focal length
scope = PolScope()
uc_mode = universal_compensator_modes(scope.swing)[0]
wavelength = scope.wavelength
focal_length = 25
NA = scope.condenser_NA
n = scope.condenser_n
diam_um = 2 * focal_length * jnp.tan(jnp.arcsin(NA / n))
res_factor = 4
shape_len = 128 * res_factor
spacing = wavelength / 2 / res_factor
pupil_field_width_um = shape_len * spacing
assert pupil_field_width_um >= diam_um, f"pupil_field_width_um ({pupil_field_width_um:.2f} um) must be >= diam_um ({diam_um:.2f} um)"
mag = 20
mla_f = scope.mla_f / mag
mla_f = 2
mla_radius = scope.mla_radius / mag
mla_separation = scope.mla_separation / mag
mla_n_y = scope.mla_n_y
mla_n_x = scope.mla_n_x
scope_short = replace(
    scope,
    shape=(shape_len, shape_len),
    condenser_f=focal_length,
    objective_f=focal_length,
    # condenser_NA=0.1,
    # objective_NA=0.05,
    spacing=spacing,
    tube_f=focal_length,
    mla_f=mla_f,
    mla_radius=mla_radius,
    mla_separation=mla_separation,
)
kohler_pt = jnp.array([0, 0])
kohler_pt = jnp.array([-7.5, 0])
# polarization = jnp.array([0.0, 0, 1.0], dtype=jnp.complex64)
polarization = uc_mode
field_output_short = scope_short.single_angle_forward_intermediates(polarization, potential_bg, kohler_pt)
# field_output_short = scope_short.single_angle_lenses(uc_mode, kohler_pt)
# field_output_short = scope_short.single_angle_illum_analyzer(polarization, kohler_pt)
# field_output_short = scope_short.forward_debug(uc_mode, potential_bg)

# %% Birefringent sample
potential_z = single_bead_sample(
    n_background=1.33,
    n_bead=jnp.array([1.45, 1.33, 1.33]),
    orientation=jnp.array([jnp.pi / 4, 0, 0]),
    radius=3.0,
    shape=(12, 256, 256),
    spacing=0.546 / 4, 
    k0=2 * jnp.pi / 0.546,
)[:-1, :-1, :-1, None]
potential = single_bead_sample(
    n_background=1.33,
    n_bead=jnp.array([1.45, 1.33, 1.33]),
    orientation=jnp.array([0, jnp.pi / 8, 0]),
    radius=3.0,
    shape=(12, 256, 256),
    spacing=0.546 / 4, 
    k0=2 * jnp.pi / 0.546,
)[:-1, :-1, :-1, None]
field_output_short = scope_short.single_angle_forward_intermediates(polarization, potential, kohler_pt)
# %% Across multiple angles
scope = PolScope()
uc_mode = universal_compensator_modes(scope.swing)[0]
wavelength = scope.wavelength
focal_length = 40
NA = scope.condenser_NA
n = scope.condenser_n
diam_um = 2 * focal_length * jnp.tan(jnp.arcsin(NA / n))
res_factor = 2
shape_len = 128 * res_factor * 2
spacing = wavelength / 2 / res_factor
pupil_field_width_um = shape_len * spacing
assert pupil_field_width_um >= diam_um, f"pupil_field_width_um ({pupil_field_width_um:.2f} um) must be >= diam_um ({diam_um:.2f} um)"
mag = 20
mla_f = scope.mla_f / mag
mla_f = 2
mla_radius = scope.mla_radius / mag
mla_separation = scope.mla_separation / mag
mla_n_y = scope.mla_n_y
mla_n_x = scope.mla_n_x
scope_short = replace(
    scope,
    shape=(shape_len, shape_len),
    condenser_f=focal_length,
    objective_f=focal_length,
    # condenser_NA=0.1,
    # objective_NA=0.05,
    spacing=spacing,
    tube_f=focal_length,
    mla_f=mla_f,
    mla_radius=mla_radius,
    mla_separation=mla_separation,
    camera_shape=(100, 100),
    camera_pitch=spacing,
)
field_output_short = scope_short.forward_debug(uc_mode, potential)
# %%
field_output_short = scope_short.single_angle_forward_intermediates(uc_mode, potential_bg, jnp.array([-7.5, 0]))
# %%
field_output_short = scope_short.single_angle_forward_intermediates(uc_mode, potential_bg, jnp.array([-7.5, 0]))
# %%
shape = (64, 64)
dx = 1.0 / 64
wavelength = 0.546
spectral_density = 1.0
waist = 1.0 / 4
amplitude = 1.0
polarization = jnp.array([0.0, 1.0j, 1.0], dtype=jnp.complex64)

# Different centers
centers = [
    (0.0, 0.0),
    (0.5 * dx, 0.5 * dx),  # slightly off a pixel center
    (8 * dx, 8 * dx),      # well within the grid
]

fields = []
for c in centers:
    field = gaussian_spot_field(
        shape=shape,
        dx=dx,
        spectrum=wavelength,
        spectral_density=spectral_density,
        waist=waist,
        center=c,
        amplitude=amplitude,
        polarization=polarization,
        phase_offset=0.0,
    )
    fields.append(field)

# Analyze each field's intensity
for center, field in zip(centers, fields):
    # field.u has shape (1, H, W, 1, 3)
    # Sum intensities over polarization dimension
    # => shape (1, H, W, 1)
    intensity_2d = jnp.sum(jnp.abs(field.u)**2, axis=-1)

    # Peak intensity on the grid
    peak_intensity = jnp.max(intensity_2d)

    # Integrated intensity (approx) => sum of intensities * pixel area
    total_intensity = jnp.sum(intensity_2d, axis=None) * (field.dx.squeeze()[0]**2)

    print(f"Center = {center}")
    print(f"  Peak intensity   = {peak_intensity:8.5f}")
    print(f"  Integrated power = {total_intensity:8.5f}")
    print()
# %% Small MLA
sample_pixels = 512
potential_bg = single_bead_sample(
    n_background=1.33,
    n_bead=jnp.array([1.33, 1.33, 1.33]),
    orientation=jnp.array([jnp.pi / 4, 0, 0]),
    radius=4.0,
    shape=(0, sample_pixels, sample_pixels),
    spacing=0.546 / 2, 
    k0=2 * jnp.pi / 0.546,
)[:-1, :-1, :-1, None]


scope = PolScope()
uc_mode = universal_compensator_modes(scope.swing)[0]
wavelength = scope.wavelength
focal_length = 40
NA = scope.condenser_NA
n = scope.condenser_n
diam_um = 2 * focal_length * jnp.tan(jnp.arcsin(NA / n))
res_factor = 2
shape_len = 128 * res_factor * 2
spacing = wavelength / 2 / res_factor
pupil_field_width_um = shape_len * spacing
assert pupil_field_width_um >= diam_um, f"pupil_field_width_um ({pupil_field_width_um:.2f} um) must be >= diam_um ({diam_um:.2f} um)"
mag = 20
mla_f = scope.mla_f / mag
mla_f = 2
mla_radius = scope.mla_radius / mag
mla_separation = scope.mla_separation / mag
mla_n_y = 3 #scope.mla_n_y
mla_n_x = 3 #scope.mla_n_x
camera_pitch = spacing
camera_pixels = int(jnp.ceil(mla_n_x * mla_radius * 2 / camera_pitch))
scope_short = replace(
    scope,
    num_angles=12,
    shape=(shape_len, shape_len),
    condenser_f=focal_length,
    objective_f=focal_length,
    condenser_NA=0.8,
    condenser_n=1.0,
    # objective_NA=0.05,
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
field_output_short = scope_short.single_angle_forward_plot(uc_mode, potential_bg, jnp.array([-35, 32]))
# field_output_short_bg = scope_short.forward_debug(uc_mode, potential_bg)
# field_output_short = scope_short.single_angle_forward_plot(uc_mode, potential_bg, jnp.array([0, 0]))

# %%
