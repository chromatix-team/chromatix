from typing import Callable

import jax.numpy as jnp
import numpy as np
from chex import assert_axis_dimension, assert_equal_shape
import matplotlib.pyplot as plt

from chromatix import Field, ScalarField, VectorField
from chromatix.typing import ArrayLike, ScalarLike
from chromatix.utils import l2_sq_norm
from chromatix.utils.shapes import (
    _broadcast_1d_to_grid,
    _broadcast_1d_to_innermost_batch,
    _broadcast_1d_to_polarization,
)

from .pupils import circular_pupil

__all__ = [
    "point_source",
    "objective_point_source",
    "plane_wave",
    "generic_field",
    "objective_kohler_illumination",
    "multi_angle_kohler_illumination",
    "single_angle_illumination",
]


# We need this alias for typing to pass
FieldPupil = Callable[[Field], Field]


def point_source(
    shape: tuple[int, int],
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike,
    z: ScalarLike,
    n: ScalarLike,
    power: ScalarLike = 1.0,
    amplitude: ScalarLike = 1.0,
    pupil: FieldPupil | None = None,
    scalar: bool = True,
    epsilon: float = float(np.finfo(np.float32).eps),
) -> ScalarField | VectorField:
    """
    Generates field due to point source a distance ``z`` away.

    Can also be given ``pupil``.

    Args:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        z: The distance of the point source.
        n: Refractive index.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarization.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
        epsilon: Value added to denominators for numerical stability.
    """
    create = ScalarField.create if scalar else VectorField.create
    # If scalar, last axis should 1, else 3.
    amplitude = jnp.atleast_1d(amplitude)
    if scalar:
        assert_axis_dimension(amplitude, -1, 1)
    else:
        assert_axis_dimension(amplitude, -1, 3)

    field = create(dx, spectrum, spectral_density, shape=shape)
    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    L = jnp.sqrt(field.spectrum * jnp.abs(z) / n)
    L_sq = jnp.sign(z) * jnp.fmax(L**2, epsilon)
    phase = jnp.pi * l2_sq_norm(field.grid) / L_sq
    u = amplitude * -1j / L_sq * jnp.exp(1j * phase)
    field = field.replace(u=u)
    if pupil is not None:
        field = pupil(field)
    return field * jnp.sqrt(power / field.power)


def objective_point_source(
    shape: tuple[int, int],
    dx: ScalarLike,
    spectrum: ScalarLike,
    z: ScalarLike,
    f: ScalarLike,
    n: ScalarLike,
    NA: ScalarLike,
    spectral_density: ScalarLike | None = None,
    power: ScalarLike | None = None,
    amplitude: ScalarLike = 1.0,
    offset: ArrayLike | tuple[float, float] = (0.0, 0.0),
) -> ScalarField | VectorField:
    """
    Generates field due to a point source defocused by an amount ``z`` away from
    the focal plane, just after passing through a lens with focal length ``f``
    and numerical aperture ``NA``.

    Args:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        z: The distance of the point source.
        f: Focal length of the objective lens.
        n: Refractive index.
        NA: The numerical aperture of the objective lens.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarization.
        offset: The offset of the point source in the plane. Should be an array
            of shape `[2,]` in the format `[y, x]`.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    spectrum = jnp.atleast_1d(spectrum)
    amplitude = jnp.atleast_1d(amplitude)

    # Equal spectral density if not given
    # NOTE: Should we add this to create?
    if spectral_density is None:
        spectral_density = jnp.full((spectrum.size,), 1 / spectrum.size)

    match amplitude.size:
        case 1:
            field = ScalarField.create(dx, spectrum, spectral_density, shape=shape)
        case 3:
            field = VectorField.create(dx, spectrum, spectral_density, shape=shape)
        case _:
            raise AssertionError("Amplitude needs to have 1 or 3 components.")

    z = _broadcast_1d_to_innermost_batch(z, field.ndim)
    offset = _broadcast_1d_to_grid(offset, field.ndim)

    L_sq = field.spectrum * f / n
    phase = -jnp.pi * (z / f) * l2_sq_norm(field.grid - offset) / L_sq
    u = amplitude * -1j / L_sq * jnp.exp(1j * phase)
    field = field.replace(u=u)
    D = 2 * f * NA / n
    field = circular_pupil(field, D)  # type: ignore

    if power is not None:
        field = field * jnp.sqrt(power / field.power)

    print(f"field.u.shape: {field.u.shape}")

    return field


def objective_kohler_illumination(
    shape: tuple[int, int],
    dx: float,
    spectrum: float | jnp.ndarray,
    f: float,
    n: float,
    NA: float,
    spectral_density: jnp.ndarray | None = None,
    power: float | None = None,
    amplitude: float | tuple[float, float, float] = 1.0,
) -> ScalarField | VectorField:
    """
    Creates a plane-wave illumination field at the sample plane, clipped by the
    objective's numerical aperture — an approximate representation of Köhler
    illumination, where the sample is uniformly illuminated.

    Args:
        shape:        (height, width) of the field grid.
        dx:           Physical spacing of grid samples (um or m).
        spectrum:     One or more wavelengths (in same length units).
        f:            Focal length of the objective lens.
        n:            Refractive index.
        NA:           Numerical aperture of the objective lens.
        spectral_density: Relative weighting for each wavelength (if multiple).
        power:        Normalize the resulting field to this total power.
        amplitude:    Amplitude of the electric field:
                      - scalar (1,) for ScalarField
                      - 3-element tuple for VectorField
    """
    # Ensure arrays
    spectrum = jnp.atleast_1d(spectrum)
    amplitude = jnp.atleast_1d(amplitude)

    # Default to equal weighting if no spectral_density is given
    if spectral_density is None:
        spectral_density = jnp.full((spectrum.size,), 1.0 / spectrum.size)

    # Create empty field (Scalar or Vector) depending on amplitude size
    match amplitude.size:
        case 1:
            field = ScalarField.create(dx, spectrum, spectral_density, shape=shape)
        case 3:
            field = VectorField.create(dx, spectrum, spectral_density, shape=shape)
        case _:
            raise ValueError("Amplitude must have 1 or 3 components (scalar or vector).")

    # -----------------------------------------------------------------
    # Step 1: Create a uniform (plane-wave) illumination in the sample plane
    # -----------------------------------------------------------------
    # Plane wave => constant phase = 0.0 (or set any phase offset you want)
    # u = amplitude * jnp.exp(1j * 0.0)
    amplitude_5d = jnp.broadcast_to(amplitude, (1, 1, 1, 1, 3))
    phase_5d     = jnp.broadcast_to(jnp.exp(1j*0.0), (1, 256, 256, 1, 1))

    u = amplitude_5d * phase_5d

    # -----------------------------------------------------------------
    # Step 2: Clip by the objective’s pupil (simulating max illumination angle)
    # -----------------------------------------------------------------
    # In Köhler, the objective’s NA ultimately limits acceptance/illumination.
    # Typically we do not put a strong "stop" on the illumination side
    # for Köhler *condenser* design, but this step mimics a real objective pupil.
    field = field.replace(u=u)
    print(f"field.u.shape: {field.u.shape}")
    D = 2.0 * f * NA / n  # pupil diameter in sample plane
    field = circular_pupil(field, D)  # type: ignore

    # -----------------------------------------------------------------
    # Step 3: (Optional) Normalize total power
    # -----------------------------------------------------------------
    if power is not None:
        current_power = field.power
        field = field * jnp.sqrt(power / current_power)

    return field


def multi_angle_kohler_illumination(
    shape: tuple[int, int],
    dx: float,
    spectrum: float | jnp.ndarray,
    f: float,
    n: float,
    NA: float,
    spectral_density: jnp.ndarray | None = None,
    power: float = 1.0,
    amplitude: float | tuple[float, float, float] = 1.0,
    num_angles: int = 10,
    incoherent: bool = True,
) -> ScalarField | VectorField:
    """
    Creates a sum (field-wise or intensity-wise) of plane waves at various
    incident angles within the numerical aperture, approximating Köhler
    illumination with polarization effects.
    """
    # Convert inputs to jax arrays
    spectrum   = jnp.atleast_1d(spectrum)
    amplitude  = jnp.atleast_1d(amplitude)

    # Default to equal weighting if not given
    if spectral_density is None:
        spectral_density = jnp.full((spectrum.size,), 1.0 / spectrum.size)

    # Create base field
    if amplitude.size == 1:
        field = ScalarField.create(dx, spectrum, spectral_density, shape=shape)
    elif amplitude.size == 3:
        field = VectorField.create(dx, spectrum, spectral_density, shape=shape)
    else:
        raise ValueError("Amplitude must be scalar (1) or 3-vector for polarization.")

    # Prepare accumulator
    if incoherent:
        accumulator = jnp.zeros_like(field.u, dtype=field.u.dtype)
    else:
        accumulator = jnp.zeros_like(field.u, dtype=field.u.dtype)

    # Retrieve (Y, X) as 2D arrays from field.grid => shape (2, H, W, 1, 1)
    # Slice off the trailing (1,1):
    Y = field.grid[0, ..., 0, 0]  # => shape (H, W)
    X = field.grid[1, ..., 0, 0]  # => shape (H, W)

    # Maximum half-angle
    theta_max = jnp.arcsin(NA / n)

    # Sample angles: radial and azimuth
    num_azimuth = 2 * num_angles
    thetas = jnp.linspace(0, theta_max, num_angles)
    phis   = jnp.linspace(0, 2*jnp.pi, num_azimuth, endpoint=False)

    # For demonstration, use only the first wavelength (or do a vmap if needed)
    lam = spectrum[0]
    k0 = (2*jnp.pi * n) / lam

    # Sum over angles
    for th in thetas:
        sin_th = jnp.sin(th)
        for ph in phis:
            kx = sin_th * jnp.cos(ph)
            ky = sin_th * jnp.sin(ph)
            # Phase factor => shape (H, W)
            phase_2d = 1j * k0 * (kx * X + ky * Y)


            if amplitude.size == 1:
                # broadcast to (H, W) => (1, H, W, 1, 1)
                phase_4d = phase_2d[..., None, None]
                phase_5d = jnp.broadcast_to(phase_4d, (*phase_2d.shape, 1, 1))
                local_u  = amplitude[0] * jnp.exp(phase_5d)
            else:
                # # vector case: define polarization for each angle, then broadcast similarly
                # # For vector field, we need to handle polarization
                # # First, broadcast phase to (1, H, W, 1, 1)
                phase_4d = phase_2d[..., None, None]
                phase_5d = jnp.broadcast_to(phase_4d, (*phase_2d.shape, 1, 1))

                lab_pol = amplitude  # shape (3,)

                # For each angle, define k = (kx, ky, kz)
                # We already have kx, ky, and we can find kz for forward-propagating waves:
                kz = jnp.sqrt((n * 2*jnp.pi / lam)**2 - (k0 * sin_th)**2) / k0
                # or just define k-vector in normalized units:
                # k_unit = (kx, ky, kz)/|k|.
                # In your code, you're using k0*(kx, ky). We have to define the sign for kz, etc.

                # Then project amplitude onto the direction orthonormal to k
                # 1) define k_unit
                k_vec     = jnp.array([kz, ky, kx])            # shape (3,)
                k_norm    = jnp.linalg.norm(k_vec)
                k_unit    = k_vec / k_norm

                # 2) dot product
                dot_val   = jnp.dot(lab_pol, k_unit)           # scalar
                # 3) subtract projection
                E_perp    = lab_pol - dot_val * k_unit         # shape (3,)

                # E_perp is now your local amplitude for that wave in the lab frame
                # If you wish to preserve amplitude magnitude strictly, you might re-normalize:
                # E_perp = E_perp * (jnp.linalg.norm(lab_pol) / (1e-16 + jnp.linalg.norm(E_perp)))

                # Expand dims for broadcasting
                E_perp_5d = E_perp.reshape(1, 1, 1, 1, 3)

                local_u = E_perp_5d * jnp.exp(phase_5d)

            if incoherent:
                accumulator += local_u.intensity
                # jnp.abs(local_u)**2
            else:
                accumulator += local_u

    # Final field
    if incoherent:
        final_u = jnp.sqrt(accumulator) * jnp.exp(1j * 0.0)
    else:
        final_u = accumulator

    new_field = field.replace(u=final_u)

    if power is not None:
        new_field = new_field * jnp.sqrt(power / new_field.power)

    # I = jnp.abs(new_field.u)**2  # or sum across polarization if vector
    I = new_field.intensity.squeeze()
    mean_I = I.mean()
    std_I = I.std()
    print("Mean intensity:", mean_I)
    print("Std intensity:", std_I)
    print("Relative uniformity:", std_I / mean_I)


    return new_field


def single_angle_illumination(shape, dx, spectrum, n, angle, amplitude=1.0, debug=False):
    """
    Returns a plane wave Field with wave vector determined by 'angle' = (theta, phi).

    We assume:
      - shape is (nz, ny, nx) for a 3D simulation.
      - field.grid is a tuple (Z, Y, X) each shaped (nz, ny, nx).
      - angle = (theta, phi), where theta is measured from the z-axis and
        phi from the x-axis in the xy-plane.
      - amplitude is a 3-element vector or scalar.
    """
    # 1) Create a base VectorField (or Field).
    field = VectorField.create(dx, spectrum, spectrum, shape=shape)

    tilted_u = make_transverse_plane_wave_local_basis(
        shape=(1, *shape),
        dx=dx,
        wavelength=spectrum,
        n_medium=n,
        theta_phi=angle,
        p_lab=amplitude,
    )

    # 7) Store in field
    field = field.replace(u=tilted_u)

    # ---- Diagnostic Plots & Tests ----
    # Let's pick z=0 slice (or the first z index if shape[0] > 1).
    z_index = 0
    # shape is (nz, ny, nx, 1, 3) or similar.
    # We'll extract each vector component: Ez, Ey, Ex
    # assuming they are in the last dimension in that order.

    Ez_slice = field.u[z_index, :, :, 0, 0]  # (ny, nx)
    Ey_slice = field.u[z_index, :, :, 0, 1]
    Ex_slice = field.u[z_index, :, :, 0, 2]

    def plot_component(comp_slice, title):
        fig, (ax_amp, ax_phi) = plt.subplots(1, 2, figsize=(10,4))
        amp = jnp.abs(comp_slice)
        phi = jnp.angle(comp_slice)

        im0 = ax_amp.imshow(amp, origin='lower', cmap='magma')
        ax_amp.set_title(f'{title} amplitude')
        plt.colorbar(im0, ax=ax_amp, fraction=0.046, pad=0.04)

        im1 = ax_phi.imshow(phi, origin='lower', cmap='twilight')
        ax_phi.set_title(f'{title} phase')
        plt.colorbar(im1, ax=ax_phi, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    if debug:
        plot_component(Ez_slice, 'Ez')
        plot_component(Ey_slice, 'Ey')
        plot_component(Ex_slice, 'Ex')

        print("Ez amplitude range:", f"{float(jnp.abs(Ez_slice).min()):.4f}", f"{float(jnp.abs(Ez_slice).max()):.4f}")
        print("Ez phase range:", f"{float(jnp.angle(Ez_slice).min()):.4f}", f"{float(jnp.angle(Ez_slice).max()):.4f}")
        print("Ey amplitude range:", f"{float(jnp.abs(Ey_slice).min()):.4f}", f"{float(jnp.abs(Ey_slice).max()):.4f}")
        print("Ey phase range:", f"{float(jnp.angle(Ey_slice).min()):.4f}", f"{float(jnp.angle(Ey_slice).max()):.4f}")
        print("Ex amplitude range:", f"{float(jnp.abs(Ex_slice).min()):.4f}", f"{float(jnp.abs(Ex_slice).max()):.4f}")
        print("Ex phase range:", f"{float(jnp.angle(Ex_slice).min()):.4f}", f"{float(jnp.angle(Ex_slice).max()):.4f}")

    # D) (Optional) 2D FFT check for each component, if desired.
    # 2D FFT check for each component
    def plot_fft2d(comp_slice, title):
        fft = jnp.fft.fftshift(jnp.fft.fft2(comp_slice))
        amp = jnp.log10(jnp.abs(fft) + 1e-10)  # Add small constant to avoid log(0)
        return amp

    if debug:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
        # Plot Ez FFT
        amp_Ez = plot_fft2d(Ez_slice, 'Ez')
        im1 = ax1.imshow(amp_Ez, origin='lower', cmap='magma')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title('Ez FFT amplitude (log scale)')
        
        # Plot Ey FFT
        amp_Ey = plot_fft2d(Ey_slice, 'Ey')
        im2 = ax2.imshow(amp_Ey, origin='lower', cmap='magma')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title('Ey FFT amplitude (log scale)')
        
        # Plot Ex FFT
        amp_Ex = plot_fft2d(Ex_slice, 'Ex')
        im3 = ax3.imshow(amp_Ex, origin='lower', cmap='magma')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title('Ex FFT amplitude (log scale)')
        
        plt.tight_layout()
        plt.show()
        #     This is a good way to confirm there's a single plane-wave peak
        #     at (kx, ky).

    return field


def cross_zyx(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Custom cross product for vectors in (z, y, x) order,
    returning the result also in (z, y, x) order.
    """
    a_z, a_y, a_x = a[..., 0], a[..., 1], a[..., 2]
    b_z, b_y, b_x = b[..., 0], b[..., 1], b[..., 2]

    # Standard cross if (x,y,z), but we re-map:
    # R_z = (a_x*b_y - a_y*b_x)
    r_z = a_x * b_y - a_y * b_x
    # R_y = (a_z*b_x - a_x*b_z)
    r_y = a_z * b_x - a_x * b_z
    # R_x = (a_y*b_z - a_z*b_y)
    r_x = a_y * b_z - a_z * b_y

    return jnp.stack([r_z, r_y, r_x], axis=-1)

def build_local_basis_zyx(k_hat: jnp.ndarray,
                          ref_vector: jnp.ndarray = jnp.array([1., 0., 0.])
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (e1, e2) in (z,y,x) order, each orthonormal to k_hat.
    
    Args:
        k_hat: shape (..., 3) with last-dim = [k_z, k_y, k_x], must be unit length
        ref_vector: shape (3,) reference for cross product to define e1
                    (also in [z,y,x] order).
    """
    # e1p = ref_vector x k_hat
    e1p = cross_zyx(ref_vector, k_hat)
    e1p_norm = jnp.linalg.norm(e1p, axis=-1, keepdims=True)

    # If ref is (nearly) parallel to k_hat, pick a fallback
    fallback_ref = jnp.array([0., 1., 0.])  # or [0,0,1], etc.
    e1p = jnp.where(
        e1p_norm < 1e-12,
        cross_zyx(fallback_ref, k_hat),
        e1p
    )
    e1 = e1p / jnp.linalg.norm(e1p, axis=-1, keepdims=True)
    e2 = cross_zyx(k_hat, e1)
    return e1, e2

def project_onto_local_basis_zyx(vec: jnp.ndarray,
                                 e1: jnp.ndarray,
                                 e2: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Projects 'vec' (shape (...,3), [z,y,x] order) onto e1, e2.
    Returns (v1, v2, v_lab_perp):
      v1, v2: the scalar complex amplitudes in the local basis
      v_lab_perp: the reconstructed vector = v1*e1 + v2*e2, shape(..., 3)
    """
    v1 = jnp.sum(vec * e1, axis=-1)  # dot product
    v2 = jnp.sum(vec * e2, axis=-1)
    # Rebuild in lab coords
    v_lab_perp = v1[..., None]*e1 + v2[..., None]*e2
    return v1, v2, v_lab_perp

def make_transverse_plane_wave_local_basis(
    shape,                   # (nz, ny, nx)
    dx,                      # float or (dz, dy, dx)
    wavelength,             # e.g. 0.546
    n_medium,               # refractive index
    theta_phi,              # (theta, phi) in radians
    p_lab,                  # user-supplied [Ez, Ey, Ex], can be complex
    desired_amp=1.0,
    ref_vector=jnp.array([1., 0., 0.])
):
    """
    Creates a 3D plane wave field E(r) = E0_perp * exp(i k · r),
    where E0_perp is the projection of p_lab onto the plane
    orthonormal to k_hat, ensuring the final wave is fully transverse
    to the direction of propagation (theta, phi).

    The last dimension is (Ez, Ey, Ex) in your (z,y,x) order.

    Args:
      shape: (nz, ny, nx)
      dx: float or (dz, dy, dx)
      wavelength: in same units as dx
      n_medium: refractive index
      theta_phi: (theta, phi)
      p_lab: shape (3,) or (Ez, Ey, Ex) in your z,y,x order
      desired_amp: total amplitude to normalize the final E to
      ref_vector: for constructing local basis via cross product
    Returns:
      E_field: shape (nz, ny, nx, 1, 3)
    """

    # 1) Angles => wavevector k_vec in (z,y,x) order
    theta, phi = theta_phi
    sin_th, cos_th = jnp.sin(theta), jnp.cos(theta)
    sin_ph, cos_ph = jnp.sin(phi), jnp.cos(phi)

    k0 = (2*jnp.pi*n_medium)/wavelength
    k_vec = k0 * jnp.array([
        cos_th,
        sin_th*sin_ph,
        sin_th*cos_ph
    ], dtype=jnp.float32)
    k_mag = jnp.linalg.norm(k_vec)
    k_hat = k_vec / k_mag

    # Build local basis e1, e2
    e1, e2 = build_local_basis_zyx(k_hat, ref_vector)
    # E0_perp = (1.0/jnp.sqrt(2)) * (e1 + 1j*e2)
    p1 = jnp.dot(p_lab, e1)  # complex scalar
    p2 = jnp.dot(p_lab, e2)  # complex scalar
    E0_perp = p1 * e1 + p2 * e2

    # Normalize amplitude
    #    "E0_perp" is shape (3,) => let's do norm along last dim
    norm_E0 = jnp.linalg.norm(E0_perp)
    # If user gave p_lab parallel to k, we might get ~0 norm
    E0_perp = jnp.where(
        norm_E0 > 1e-12,
        (desired_amp/norm_E0)*E0_perp,
        jnp.array([desired_amp,0,0])  # or some fallback
    )

    # Build the coordinate arrays
    if isinstance(dx, (tuple, list)):
        dz, dy, dx_ = dx
    else:
        dz = dy = dx_ = dx
    nz, ny, nx = shape

    z_vals = (jnp.arange(nz) - nz//2) * dz
    y_vals = (jnp.arange(ny) - ny//2) * dy
    x_vals = (jnp.arange(nx) - nx//2) * dx_
    Z, Y, X = jnp.meshgrid(z_vals, y_vals, x_vals, indexing='ij')

    # Phase factor exp(i k · r) => k_z * Z + k_y * Y + k_x * X
    phase_3d = 1j*(k_vec[0]*Z + k_vec[1]*Y + k_vec[2]*X)
    phase_factor = jnp.exp(phase_3d)

    # Multiply by E0_perp => shape (nz, ny, nx, 3)
    E_field = phase_factor[..., None] * E0_perp

    # Insert "channel" dim => final shape (nz, ny, nx, 1, 3)
    E_field = E_field[..., None, :]

    return E_field


def make_transverse_plane_wave(
    shape,          # (nz, ny, nx)
    dx,             # (dz, dy, dx) or single float
    wavelength,     # e.g. in same units as dx
    n_medium,       # refractive index
    theta_phi,      # (theta, phi) in radians
    p_lab,          # (px, py, pz) initial polarization in lab coordinates
    desired_amp=1.0
):
    """
    Creates a 3D plane wave field E(r) = E0_perp * exp(i k · r),
    ensuring that E0_perp is perpendicular to k, but oriented
    'as close as possible' to p_lab (the initial user-specified polarization).
    
    The final amplitude is normalized to 'desired_amp'.
    Returns a field array shaped (nz, ny, nx, 1, 3) with components (Ez, Ey, Ex).
    """

    # 1. Unpack angles
    theta, phi = theta_phi
    sin_th = jnp.sin(theta)
    cos_th = jnp.cos(theta)
    sin_ph = jnp.sin(phi)
    cos_ph = jnp.cos(phi)

    # 2. Construct wave vector k = k0 * [sin_th cos_ph, sin_th sin_ph, cos_th]
    k0 = (2 * jnp.pi * n_medium) / wavelength
    # k_vec = k0 * jnp.array([sin_th * cos_ph, sin_th * sin_ph, cos_th])
    k_vec = k0 * jnp.array([cos_th, sin_th * sin_ph, sin_th * cos_ph])

    # 3. Project p_lab onto plane perpendicular to k
    p_lab = jnp.array(p_lab, dtype=jnp.complex64)  # ensure complex friendly
    dot_pk = jnp.dot(p_lab, k_vec)
    k_mag_sq = jnp.dot(k_vec, k_vec)  # = (k0^2)
    E0_perp = p_lab - (dot_pk / k_mag_sq) * k_vec

    # 4. Normalize to desired_amp
    norm_E0 = jnp.linalg.norm(E0_perp)
    # If p_lab was parallel to k, norm_E0 might be zero. Check / handle that:
    E0_perp = jnp.where(norm_E0 > 1e-12, (desired_amp / norm_E0) * E0_perp, jnp.array([desired_amp, 0, 0]))

    # 5. Build coordinate arrays
    if isinstance(dx, (tuple, list)):
        dz, dy, dx_ = dx
    else:
        # assume dx is uniform spacing for z,y,x
        dz = dy = dx_ = dx
    nz, ny, nx = shape

    # Create Z, Y, X: each shaped (nz, ny, nx)
    z_vals = (jnp.arange(nz) - (nz//2)) * dz
    y_vals = (jnp.arange(ny) - (ny//2)) * dy
    x_vals = (jnp.arange(nx) - (nx//2)) * dx_
    Z, Y, X = jnp.meshgrid(z_vals, y_vals, x_vals, indexing='ij')

    # 6. Compute exp(i k · r)
    #    r = (x, y, z), but note the ordering difference
    #    k · r = kx X + ky Y + kz Z
    # phase_3d = 1j * (k_vec[0] * X + k_vec[1] * Y + k_vec[2] * Z)
    phase_3d = 1j * (k_vec[2] * X + k_vec[1] * Y + k_vec[0] * Z)
    phase_factor = jnp.exp(phase_3d)

    # 7. Multiply the entire 3D array by E0_perp
    #    We want shape: (nz, ny, nx, 1, 3)
    #    E0_perp is shape (3, ), so broadcast
    E_field = phase_factor[..., None] * E0_perp  # shape (nz, ny, nx, 3)

    E0_numeric = E_field[0,0,0]   # e.g. the constant amplitude factor at the center
    
    dot_val = jnp.dot(E0_numeric, k_vec)
    if False:
        print(f"E0_numeric: {E0_numeric}, E0 shape: {E0_numeric.shape}")
        print("E dot k =", dot_val.real, dot_val.imag)


    # Insert a "channel" dimension or something if needed
    # final shape => (nz, ny, nx, 1, 3)
    E_field = E_field[..., None, :]  # just to match certain frameworks

    if False:
        print(f"E_field.shape: {E_field.shape}")
        print(f"E_field[0,0,0]: {E_field[0,0,0]}")

    return E_field


def single_angle_illumination_try1(shape, dx, spectrum, n, angle, amplitude=1.0):
    """
    Returns a plane wave Field with wave vector determined by 'angle' = (theta, phi).
    
    This version includes the z-component of the wave vector. We assume:
      - shape is (nz, ny, nx) for a 3D simulation.
      - field.grid is a tuple (Z, Y, X) each shaped (nz, ny, nx).
      - angle = (theta, phi), where theta is measured from the z-axis and phi from x-axis in the xy-plane.

    Parameters
    ----------
    shape : tuple of int
        (nz, ny, nx) shape of the simulation domain.
    dx : float or tuple
        Grid spacing (dz, dy, dx) in the same units as 'spectrum' wavelength.
    spectrum : float
        Wavelength (lambda) in the medium, or vacuum wavelength if 'n' is used for index.
    n : float
        Refractive index of the medium in which the wave is propagating.
    angle : (theta, phi)
        theta = polar angle from z-axis
        phi   = azimuthal angle in x-y plane
    amplitude : float
        Scalar amplitude for the wave.

    Returns
    -------
    field : Field
        A 3D Field object whose complex amplitude is a plane wave traveling
        at the specified angles. 
    """
    # 1) Create a base VectorField (or Field) 
    field = VectorField.create(dx, spectrum, spectrum, shape=shape)

    # 2) Wave number in the medium
    lam = spectrum
    k0 = (2 * jnp.pi * n) / lam  # magnitude of the wave vector

    # Unpack angles
    theta, phi = angle
    print(f"theta: {theta}, phi: {phi}")

    # 3) Components of the unit direction vector in spherical coords:
    #    Using the physics convention: 
    #        kx = sin(theta)*cos(phi)
    #        ky = sin(theta)*sin(phi)
    #        kz = cos(theta)
    sin_th = jnp.sin(theta)
    cos_th = jnp.cos(theta)
    cos_ph = jnp.cos(phi)
    sin_ph = jnp.sin(phi)

    kx = sin_th * cos_ph
    ky = sin_th * sin_ph
    kz = cos_th

    print(f"kx: {kx}, ky: {ky}, kz: {kz}")

    # 4) Coordinates: assume field.grid = (Z, Y, X) 
    #    Each is shaped (nz, ny, nx)
    Z = field.grid[0]
    Y = field.grid[1]
    X = field.grid[2]

    # 5) Compute phase. 
    #    Full 3D plane wave: exp(i * k . r) = exp(i*k0*(kx*X + ky*Y + kz*Z))
    phase_3d = 1j * k0 * (kx * X + ky * Y + kz * Z)

    # 6) Multiply by amplitude
    tilted_u = amplitude * jnp.exp(phase_3d)
    print(f"tilted_u.shape: {tilted_u.shape}, tilted_u[0, :, :, 0, 0].shape: {tilted_u[0, :, :, 0, 0].shape}")

    phi = jnp.angle(tilted_u[0, :, :, 0, 2])
    plt.imshow(phi, cmap='twilight')
    plt.colorbar()
    plt.show()


    field = field.replace(u=tilted_u)

    return field


def single_angle_illumination_og(
    shape, dx, spectrum, n, angle, amplitude=1.0
) -> Field:
    """
    Returns a plane wave Field tilted by 'angle' (theta, phi, etc.).
    """
    # 1) Create a base Field (probably scalar for simplicity)
    field = VectorField.create(dx, spectrum, spectrum, shape=shape)

    # 2) Insert the plane-wave tilt
    #    angle might be (theta, phi) or (kx, ky) ...
    #    Suppose angle = (theta, phi)
    lam = spectrum
    k0 = (2*jnp.pi * n) / lam
    theta, phi = angle

    # define the tilt
    sin_th = jnp.sin(theta)
    kx = sin_th * jnp.cos(phi)
    ky = sin_th * jnp.sin(phi)

    # coordinates
    Y = field.grid[0, ..., 0, 0]
    X = field.grid[1, ..., 0, 0]

    phase_2d = 1j * k0 * (kx * X + ky * Y)

    # broadcast to the full shape
    phase_4d = phase_2d[..., None, None]
    tilted_u = amplitude * jnp.exp(phase_4d)

    # embed in field
    field = field.replace(u=tilted_u)

    return field


def plane_wave(
    shape: tuple[int, int],
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike | None = None,
    power: ScalarLike | None = None,
    amplitude: ScalarLike = 1.0,
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0),
    pupil: FieldPupil | None = None,
) -> ScalarField | VectorField:
    """
    Generates plane wave of given ``power``.

    Can also be given ``pupil`` and ``kykx`` vector to control the angle of the
    plane wave.

    Args:
        shape: The shape (height and width) of the ``Field`` to be created.
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        power: The total power that the result should be normalized to,
            defaults to 1.0
        amplitude: The amplitude of the electric field. For ``ScalarField`` this
            doesnt do anything, but it is required for ``VectorField`` to set
            the polarization.
        kykx: Defines the orientation of the plane wave. Should be an
            array of shape `[2,]` in the format `[ky, kx]`.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """

    spectrum = jnp.atleast_1d(spectrum)
    amplitude = jnp.atleast_1d(amplitude)

    # Equal spectral density if not given
    # NOTE: Should we add this to create?
    if spectral_density is None:
        spectral_density = jnp.full((spectrum.size,), 1 / spectrum.size)

    match amplitude.size:
        case 1:
            field = ScalarField.create(dx, spectrum, spectral_density, shape=shape)
        case 3:
            field = VectorField.create(dx, spectrum, spectral_density, shape=shape)
        case _:
            raise AssertionError("Amplitude needs to have 1 or 3 components.")

    # Setting field
    # NOTE: Setting grid and amplitude to the final dim would get rid of this
    kykx = _broadcast_1d_to_grid(kykx, field.ndim)
    amplitude = _broadcast_1d_to_polarization(amplitude, field.ndim)
    u = amplitude * jnp.exp(1j * jnp.sum(kykx * field.grid, axis=0))
    field = field.replace(u=u)

    # Adding pupil and power, if necessaary
    if pupil is not None:
        field = pupil(field)

    if power is not None:
        field = field * jnp.sqrt(power / field.power)

    return field


def generic_field(
    dx: ScalarLike,
    spectrum: ScalarLike,
    spectral_density: ScalarLike,
    amplitude: ArrayLike,
    phase: ArrayLike,
    power: ScalarLike = 1.0,
    pupil: FieldPupil | None = None,
    scalar: bool = True,
) -> ScalarField | VectorField:
    """
    Generates field with arbitrary ``phase`` and ``amplitude``.

    Can also be given ``pupil``.

    Args:
        dx: The spacing of the samples of the ``Field``.
        spectrum: The wavelengths included in the ``Field`` to be created.
        spectral_density: The weights of each wavelength in the ``Field`` to
            be created.
        amplitude: The amplitude of the field with shape `(B... H W C [1 | 3])`.
        phase: The phase of the field with shape `(B... H W C [1 | 3])`.
        power: The total power that the result should be normalized to,
            defaults to 1.0.
        pupil: If provided, will be called on the field to apply a pupil.
        scalar: Whether the result should be ``ScalarField`` (if True) or
            ``VectorField`` (if False). Defaults to True.
    """
    create = ScalarField.create if scalar else VectorField.create
    assert (
        amplitude.ndim >= 5
    ), "Amplitude must have at least 5 dimensions: (B... H W C [1 | 3])"
    assert (
        phase.ndim >= 5
    ), "Phase must have at least 5 dimensions: (B... H W C [1 | 3])"
    vectorial_dimension = 1 if scalar else 3
    assert_axis_dimension(amplitude, -1, vectorial_dimension)
    assert_axis_dimension(phase, -1, vectorial_dimension)
    assert_equal_shape([amplitude, phase])
    u = jnp.array(amplitude) * jnp.exp(1j * phase)
    field = create(dx, spectrum, spectral_density, u=u)
    if pupil is not None:
        field = pupil(field)
    return field * jnp.sqrt(power / field.power)
