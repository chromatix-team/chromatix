# %% helpers.py

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt

import chromatix.functional as cf

def ret_and_azim_from_intensity(image_list, swing, debug=False):
    """
    Calculate retardance and azimuth using the PolScope 5-frame algorithm.
    """
    if len(image_list) != 5:
        raise ValueError(f"Expected 5 images, got {len(image_list)}.")
    
    a = image_list[4] - image_list[1]
    b = image_list[2] - image_list[3]
    if debug:
        # Plot them
        plt.imshow(a, cmap='gray'); plt.title("a = I4 - I3"); plt.colorbar(); plt.show()
        plt.imshow(b, cmap='gray'); plt.title("b = I2 - I1"); plt.colorbar(); plt.show()
    den = (image_list[1] + image_list[2] + image_list[3] + image_list[4] - 4 * image_list[0]) / 2
    if debug:
        plt.figure(figsize=(6,4))
        plt.imshow(den, cmap='gray')
        plt.colorbar(label='den')
        plt.title('Denominator')
        plt.show()
    # Check where denominator is close to zero
    den_is_zero = np.isclose(den, 0, rtol=1e-10, atol=1e-6)
    if np.any(den_is_zero):
        print(f"Found {np.sum(den_is_zero)} pixels where denominator is close to zero")
        if debug:
            # Optional: Plot locations where den ≈ 0
            plt.figure(figsize=(6,4))
            plt.imshow(den_is_zero, cmap='gray')
            plt.colorbar(label='den ≈ 0')
            plt.title('Pixels where denominator is close to zero')
            plt.show()
    prefactor = np.tan(swing / 2)
    tmp = np.arctan(prefactor * np.sqrt(a**2 + b**2) / (np.abs(den) + np.finfo(float).eps))
    if debug:
        plt.figure(figsize=(6,4))
        plt.imshow(tmp, cmap='gray')
        plt.colorbar(label='tmp')
        plt.title('tmp')
        plt.show()
    ret = np.where(den == 0, np.pi / 2, tmp)
    ret = np.where(den < 0, np.pi - tmp, ret)

    azim_val = (0.5 * (np.arctan2(-a / 2, b) + np.pi)) % np.pi
    # azim_val = (azim_val + np.pi/4) % np.pi
    azim = np.where((a == 0) & (b == 0), 0, azim_val)
    # azim = np.where((a == 0) & (b == 0), 0, (0.5 * (np.arctan2(-a / 2, b) + np.pi)) % np.pi)
    

    amp = np.sqrt(a**2 + b**2)
    small_threshold = 1e-7
    azim = np.where(amp < small_threshold, 0, azim)
    ret  = np.where(amp < small_threshold, 0, ret)
    return [ret, azim]


def ret_and_azim_from_intensity_og(image_list, swing):
    """
    Calculate retardance and azimuth using the PolScope 5-frame algorithm.
    """
    if len(image_list) != 5:
        raise ValueError(f"Expected 5 images, got {len(image_list)}.")
    
    a = image_list[4] - image_list[1]
    b = image_list[2] - image_list[3]
    den = (image_list[1] + image_list[2] + image_list[3] + image_list[4] - 4 * image_list[0]) / 2
    prefactor = np.tan(swing / 2)
    tmp = np.arctan(prefactor * np.sqrt(a**2 + b**2) / (np.abs(den) + np.finfo(float).eps))
    ret = np.where(den == 0, np.pi / 2, tmp)
    ret = np.where(den < 0, np.pi - tmp, ret)
    azim = np.where((a == 0) & (b == 0), 0, (0.5 * (np.arctan2(-a / 2, b) + np.pi)) % np.pi)
    return [ret, azim]


def ret_and_azim_from_intensity_with_bg(sample_images, bg_images, swing):
    """
    Compute retardance and azimuth from a 5-frame PolScope setup,
    with background subtraction, using np.isclose checks.

    Parameters
    ----------
    sample_images : list of 5 numpy arrays
        The raw intensity images [I0, I1, I2, I3, I4] with the SAMPLE in place.
    bg_images : list of 5 numpy arrays
        The raw intensity images [I0b, I1b, I2b, I3b, I4b] taken with NO sample.
    swing : float
        The known compensator swing (in radians) used in the PolScope.

    Returns
    -------
    ret : 2D numpy array
        The background-corrected retardance map.
    azim : 2D numpy array
        The background-corrected azimuth (slow-axis orientation) map.
    """
    if len(sample_images) != 5 or len(bg_images) != 5:
        raise ValueError("Expected 5 images each for sample and background.")

    eps = np.finfo(float).eps
    prefactor = np.tan(swing / 2)
    
    # Helper to compute the PolScope differences for any 5-frame set
    def polscope_diffs(imgs):
        # a = I4 - I1
        a_ = imgs[4] - imgs[1]
        # b = I2 - I3
        b_ = imgs[2] - imgs[3]
        # den = (I1 + I2 + I3 + I4 - 4*I0) / 2
        den_ = (imgs[1] + imgs[2] + imgs[3] + imgs[4] - 4 * imgs[0]) / 2
        return a_, b_, den_

    # 1) Get differences for sample and for background
    a_s, b_s, den_s = polscope_diffs(sample_images)
    a_b, b_b, den_b = polscope_diffs(bg_images)

    # Debug/diagnostic plots of the difference images:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(a_s - a_b, cmap='gray')
    axes[0].set_title("a_obj")
    axes[0].set_axis_off()
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(b_s - b_b, cmap='gray')
    axes[1].set_title("b_obj")
    axes[1].set_axis_off()
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(den_s - den_b, cmap='gray')
    axes[2].set_title("den_obj")
    axes[2].set_axis_off()
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

    # 2) Subtract background differences from sample differences
    a_obj = a_s - a_b
    b_obj = b_s - b_b
    den_obj = den_s - den_b

    # 3) Convert those corrected differences into retardance, azimuth
    # same logic as original ret_and_azim_from_intensity(), but with isclose
    amp = np.sqrt(a_obj**2 + b_obj**2)   # amplitude for threshold checks

    # Add a small offset to denom to avoid divide-by-zero
    denom = np.abs(den_obj) + eps

    numer = prefactor * amp
    tmp = np.arctan(numer / denom)

    # We'll define a small tolerance for "close to zero" checks
    # This might need tweaking depending on your noise floor
    amp_tol  = 1e-6
    den_tol  = 1e-6

    # If den_obj is effectively zero => set ret to pi/2
    # (this is the standard PolScope approach if the denominator is 0)
    zero_den_mask = np.isclose(den_obj, 0.0, atol=den_tol)
    ret = np.where(zero_den_mask, np.pi/2, tmp)

    # If den_obj < 0, we do pi - tmp (for typical PolScope sign flipping).
    # But only do this if the amplitude is above the noise threshold.
    # Otherwise, we'll just keep ret=0 if the amplitude is near zero.
    neg_den_mask = (den_obj < 0) & ~zero_den_mask & (amp > amp_tol)
    ret = np.where(neg_den_mask, np.pi - tmp, ret)

    # If amplitude is below threshold, force ret=0
    zero_amp_mask = np.isclose(amp, 0.0, atol=amp_tol)
    ret = np.where(zero_amp_mask, 0, ret)

    # Azimuth calculation: arctan2( -a/2, b ) => half of that plus pi offset
    # If a_obj and b_obj are both ~0 => azim=0
    ab_zero_mask = np.isclose(a_obj, 0.0, atol=amp_tol) & \
                   np.isclose(b_obj, 0.0, atol=amp_tol)

    azim_raw = 0.5 * (np.arctan2(-a_obj / 2.0, b_obj) + np.pi)
    azim = np.where(ab_zero_mask, 0.0, azim_raw % np.pi)

    return ret, azim


def generate_mla_mask(camera_shape, camera_pitch, mla_n_y, mla_n_x, mla_separation, mla_radius):
    """
    Generate a microlens array (MLA) mask for camera images.
    """
    N = camera_shape[0]
    s = camera_pitch
    det_coord = np.linspace(-N * s / 2, N * s / 2, N) + s / 2
    XX, YY = np.meshgrid(det_coord, det_coord)

    # Define MLA coordinates
    unit_coords = np.meshgrid(
        np.arange(mla_n_y) - mla_n_y // 2,
        np.arange(mla_n_x) - mla_n_x // 2,
        indexing="ij",
    )
    unit_coords = np.array(unit_coords).reshape(2, -1)
    x_mla, y_mla = unit_coords * mla_separation

    # Create a vectorized circular mask for the MLA
    r_squared = mla_radius ** 2
    x_mla = x_mla[:, None, None]
    y_mla = y_mla[:, None, None]
    distances = (XX[np.newaxis, :, :] - x_mla) ** 2 + (YY[np.newaxis, :, :] - y_mla) ** 2
    mask = np.any(distances < r_squared, axis=0)
    return mask


def plot_propagated_field(field, title=None, vmin=None, vmax=None, presquared=False):
    """Plots the intensity and the x, y, z components of the field amplitude."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    # Plot intensity
    intensity_data = field.intensity.squeeze()
    if intensity_data.size == 1:
        intensity_data = np.array([[intensity_data]])
    im_intensity = axes[0].imshow(intensity_data, cmap='inferno')
    axes[0].set_title("Intensity", fontsize=14)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im_intensity, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Extract amplitude components - handle different shapes
    # Check field.amplitude shape to determine how to extract components
    if field.amplitude.ndim >= 5:
        # Standard case with [batch, z, y, x, component]
        if presquared:
            data_ex = field.amplitude[0, :, :, 0, 2].squeeze()
            data_ey = field.amplitude[0, :, :, 0, 1].squeeze()
            data_ez = field.amplitude[0, :, :, 0, 0].squeeze()
        else:
            data_ex = field.amplitude[0, :, :, 0, 2].squeeze() ** 2
            data_ey = field.amplitude[0, :, :, 0, 1].squeeze() ** 2
            data_ez = field.amplitude[0, :, :, 0, 0].squeeze() ** 2
    else:
        # Handle case where field.amplitude is just a vector [z, y, x]
        if presquared:
            data_ex = field.amplitude[2]
            data_ey = field.amplitude[1]
            data_ez = field.amplitude[0]
        else:
            data_ex = field.amplitude[2] ** 2 # x-component
            data_ey = field.amplitude[1] ** 2 # y-component
            data_ez = field.amplitude[0] ** 2 # z-component

    # Handle scalar values by converting to 2D arrays
    if data_ex.size == 1:
        data_ex = np.array([[data_ex]])
    if data_ey.size == 1:
        data_ey = np.array([[data_ey]])
    if data_ez.size == 1:
        data_ez = np.array([[data_ez]])

    if vmin is None:
        vmin = min(data_ex.min(), data_ey.min(), data_ez.min())
    if vmax is None:
        vmax = max(data_ex.max(), data_ey.max(), data_ez.max())
    
    # Plot each electric field component
    im_ex = axes[1].imshow(data_ex, vmin=vmin, vmax=vmax, cmap='inferno')
    axes[1].set_title("$|E_x|^2$", fontsize=14)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    im_ey = axes[2].imshow(data_ey, vmin=vmin, vmax=vmax, cmap='inferno')
    axes[2].set_title("$|E_y|^2$", fontsize=14)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    im_ez = axes[3].imshow(data_ez, vmin=vmin, vmax=vmax, cmap='inferno')
    axes[3].set_title("$|E_z|^2$", fontsize=14)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    # Add a common colorbar
    fig.subplots_adjust(right=0.9)
    pos = axes[1].get_position()
    cbar_ax = fig.add_axes([0.92, pos.y0, 0.01, pos.height])
    fig.colorbar(im_ex, cax=cbar_ax)
    
    plt.show()
    return fig


def plot_camera_images(images):
    """Plots a row of camera images in grayscale with a single colorbar."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(16, 4), tight_layout=True)
    
    im = None
    for i, (ax, img) in enumerate(zip(axes, images)):
        im = ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$\\Sigma_{{{i+1}}}$", fontsize=16)
    
    plt.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.show(block=True)
    return fig


def plot_retardance_azimuth(ret, azim):
    """Plots the Retardance and Azimuth images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("PolScope Images", fontsize=16)

    im_ret = axes[0].imshow(ret, cmap="plasma")
    axes[0].set_title("Retardance", fontsize=14)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im_ret, ax=axes[0], fraction=0.046, pad=0.04)

    im_azim = axes[1].imshow(azim, cmap="twilight")
    axes[1].set_title("Azimuth", fontsize=14)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im_azim, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=True)
    return fig


def plot_propagated_field_separate_colorbars(field, title=None, presquared=False):
    """Plots the intensity and the x, y, z components of the field amplitude with separate colorbars."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    # Plot intensity
    intensity_data = field.intensity.squeeze()
    if intensity_data.size == 1:
        intensity_data = np.array([[intensity_data]])
    im_intensity = axes[0].imshow(intensity_data, cmap='inferno')
    axes[0].set_title("Intensity", fontsize=14)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im_intensity, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Extract amplitude components - handle different shapes
    # Check field.amplitude shape to determine how to extract components
    if field.amplitude.ndim >= 5:
        # Standard case with [batch, z, y, x, component]
        if presquared:
            data_ex = field.amplitude[0, :, :, 0, 2].squeeze()
            data_ey = field.amplitude[0, :, :, 0, 1].squeeze()
            data_ez = field.amplitude[0, :, :, 0, 0].squeeze()
        else:
            data_ex = field.amplitude[0, :, :, 0, 2].squeeze() ** 2
            data_ey = field.amplitude[0, :, :, 0, 1].squeeze() ** 2
            data_ez = field.amplitude[0, :, :, 0, 0].squeeze() ** 2
    else:
        # Handle case where field.amplitude is just a vector [z, y, x]
        if presquared:
            data_ex = field.amplitude[2]
            data_ey = field.amplitude[1]
            data_ez = field.amplitude[0]
        else:
            data_ex = field.amplitude[2] ** 2  # x-component
            data_ey = field.amplitude[1] ** 2  # y-component
            data_ez = field.amplitude[0] ** 2  # z-component

    # Handle scalar values by converting to 2D arrays
    if data_ex.size == 1:
        data_ex = np.array([[data_ex]])
    if data_ey.size == 1:
        data_ey = np.array([[data_ey]])
    if data_ez.size == 1:
        data_ez = np.array([[data_ez]])

    # Plot each electric field component with individual scales
    im_ex = axes[1].imshow(data_ex, cmap='inferno')
    axes[1].set_title("$|E_x|^2$", fontsize=14)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im_ex, ax=axes[1], fraction=0.046, pad=0.04)

    im_ey = axes[2].imshow(data_ey, cmap='inferno')
    axes[2].set_title("$|E_y|^2$", fontsize=14)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im_ey, ax=axes[2], fraction=0.046, pad=0.04)

    im_ez = axes[3].imshow(data_ez, cmap='inferno')
    axes[3].set_title("$|E_z|^2$", fontsize=14)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im_ez, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_field_phase(field, title=None, phase_cmap='twilight'):
    """Plots the intensity and the x, y, z components of the field phase with separate colorbars."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    # Plot intensity
    intensity_data = field.intensity.squeeze()
    if intensity_data.size == 1:
        intensity_data = np.array([[intensity_data]])
    im_intensity = axes[0].imshow(intensity_data, cmap='inferno')
    axes[0].set_title("Intensity", fontsize=14)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im_intensity, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Extract amplitude components - handle different shapes
    # Check field.amplitude shape to determine how to extract components
    if field.amplitude.ndim >= 5:
        # Standard case with [batch, z, y, x, component]
        phase_ex = np.angle(field.amplitude[0, :, :, 0, 2].squeeze())
        phase_ey = np.angle(field.amplitude[0, :, :, 0, 1].squeeze())
        phase_ez = np.angle(field.amplitude[0, :, :, 0, 0].squeeze())
    else:
        # Handle case where field.amplitude is just a vector [z, y, x]
        phase_ex = np.angle(field.amplitude[2])  # x-component
        phase_ey = np.angle(field.amplitude[1])  # y-component
        phase_ez = np.angle(field.amplitude[0])  # z-component

    # Handle scalar values by converting to 2D arrays
    if np.isscalar(phase_ex) or phase_ex.size == 1:
        phase_ex = np.array([[phase_ex]])
    if np.isscalar(phase_ey) or phase_ey.size == 1:
        phase_ey = np.array([[phase_ey]])
    if np.isscalar(phase_ez) or phase_ez.size == 1:
        phase_ez = np.array([[phase_ez]])

    # Plot each electric field phase component
    im_ex = axes[1].imshow(phase_ex, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase $E_x$", fontsize=14)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im_ex, ax=axes[1], fraction=0.046, pad=0.04)

    im_ey = axes[2].imshow(phase_ey, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    axes[2].set_title("Phase $E_y$", fontsize=14)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im_ey, ax=axes[2], fraction=0.046, pad=0.04)

    im_ez = axes[3].imshow(phase_ez, cmap=phase_cmap, vmin=-np.pi, vmax=np.pi)
    axes[3].set_title("Phase $E_z$", fontsize=14)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    fig.colorbar(im_ez, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    return fig


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


def apply_jones_in_lab_basis(field, M_lab_2x2):
    """
    Applies a 2×2 Jones matrix defined in the lab basis to the Field object,
    assuming the field.u has shape (..., 3) in the order (z, y, x).

    We embed the 2×2 matrix into a 3×3 block that leaves the z-component unchanged:
    
      M_lab_3x3 = [[ 1     0           0          ],
                   [ 0   (m_{yy})   (m_{yx})    ],
                   [ 0   (m_{xy})   (m_{xx})    ]]

    Then we multiply field.u by M_lab_3x3^T (because each row is a 3-vector).

    Parameters
    ----------
    field : Field
        An object with an attribute `field.u` of shape (..., 3),
        where the last dimension is (z, y, x).
    M_lab_2x2 : np.ndarray, shape (2, 2)
        A Jones matrix acting on the (y, x) subspace in the lab frame.

    Returns
    -------
    field_out : Field
        A new Field (or a modified one) with updated electric field.
    """

    # 1) Embed the 2×2 lab-basis matrix into a 3×3 that leaves z alone.
    #    Because the order is (z, y, x), we place M_lab_2x2 in the bottom-right 2×2 block:
    M_lab_3x3 = np.eye(3, dtype=M_lab_2x2.dtype)
    M_lab_3x3[1:, 1:] = M_lab_2x2  # sub-block for (y, x)

    # 2) Multiply field.u by M_lab_3x3^T.
    E_in = field.u  # shape (..., 3)
    E_out = E_in @ M_lab_3x3.T  # row-vector times matrix => broadcast along leading dims

    # 3) Return a new Field (or modify in-place, depending on your code style).
    field_out = field.replace(u=E_out)
    return field_out


def apply_jones_in_wave_basis(field, M_wave_2x2, n_medium, spectrum, angle):
    """
    Applies a wave-basis Jones matrix (2x2) to the Field object in the lab frame.

    This function:
      1) Computes the wavevector via wavevector_from_angle(n_medium, spectrum, angle).
      2) Converts the 2x2 wave-basis matrix M_wave_2x2 into a 3x3 lab-basis matrix M_lab_3x3.
      3) Multiplies field.u by M_lab_3x3 along the last axis.

    Parameters
    ----------
    field : Field
        An object with an attribute `field.u` of shape (..., 3).
        The last dimension of size 3 corresponds to (E_z, E_y, E_x) in the lab frame.
    M_wave_2x2 : np.ndarray
        A 2×2 Jones matrix defined in the (s, p) wave basis.
    n_medium : float
        Refractive index of the medium.
    spectrum : float or np.ndarray
        Wavelength (or array of wavelengths). Adjust according to your setup.
    angle : tuple[float, float]
        Angle of incidence (theta, phi) in radians.

    Returns
    -------
    field_out : Field
        A new Field instance (or possibly the same instance if desired),
        whose electric field has been transformed by M_wave_2x2 in the wave basis.
    """

    # Obtain the wavevector from a user-defined function
    k_vec, k_mag, k_hat = wavevector_from_angle(n_medium, spectrum, angle)
    #  -> k_vec is typically shape (3,) in your (z, y, x) ordering

    # Convert the 2x2 wave-basis matrix into a 3x3 lab-basis matrix
    M_lab_3x3 = jones_matrix_wave_to_lab(M_wave_2x2, k_vec, reference_axis_lab=jnp.array([1, 0, 0]))

    # Apply this 3×3 matrix to the last dimension of field.u
    #    If field.u has shape (..., 3), then (..., 3) @ (3, 3) => (..., 3)
    E_in = field.u  # shape (..., 3)
    
    # Reshape E_in to handle arbitrary batch dimensions
    original_shape = E_in.shape
    E_in_flat = E_in.reshape(-1, 3)  # Flatten all but the last dimension
    
    # Apply the matrix to each 3-vector
    E_out_flat = E_in_flat @ M_lab_3x3.T  # Matrix multiplication along last axis
    
    # Reshape back to original shape
    E_out = E_out_flat.reshape(original_shape)

    field_out = field.replace(u=E_out)

    return field_out


def wavevector_from_angle(
    n_medium: float,
    wavelength: float,
    angle: tuple[float, float],
    high_precision: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generates the wavevector k in (z, y, x) order, given:
      - n_medium: refractive index
      - wavelength: in same units as angle spacing
      - angle: (theta, phi) in radians
    
    Returns:
      k_vec: shape (3,), [k_z, k_y, k_x]
      k_mag: scalar, norm of k_vec
      k_hat: shape (3,), normalized wavevector
    """
    if high_precision:
        dtype_float = jnp.float64
    else:
        dtype_float = jnp.float32

    theta, phi = angle
    sin_th = jnp.sin(theta)
    cos_th = jnp.cos(theta)
    sin_ph = jnp.sin(phi)
    cos_ph = jnp.cos(phi)

    # base magnitude k0 = 2πn / λ
    k0 = (2 * jnp.pi * n_medium) / wavelength

    # Build wavevector in the (z, y, x) ordering
    k_vec = k0 * jnp.array([
        cos_th,
        sin_th * sin_ph,
        sin_th * cos_ph
    ], dtype=dtype_float)

    # Compute magnitude and unit vector
    k_mag = jnp.linalg.norm(k_vec)
    k_hat = k_vec / k_mag

    return k_vec, k_mag, k_hat


def jones_matrix_wave_to_lab(
    M_wave_2x2: jnp.ndarray,
    k_lab: jnp.ndarray,
    reference_axis_lab: jnp.ndarray = jnp.array([1.0, 0.0, 0.0]),
    fallback_axes_lab: jnp.ndarray = None,
    tol: float = 1e-12,
    high_precision: bool = False,
    debug: bool = False
) -> jnp.ndarray:
    """
    JIT-compatible version of jones_matrix_wave_to_lab.

    Embeds a 2×2 Jones matrix (acting on s–p wave basis) into a 3×3 matrix 
    in the lab coordinate system [z, y, x]. This 3×3 acts on 3D E_lab = [E_z, E_y, E_x],
    leaving any component along k-hat unchanged (transverse wave assumption).

    Parameters
    ----------
    M_wave_2x2 : jnp.ndarray, shape (2, 2)
        The Jones matrix in the (s, p) basis.
    k_lab : jnp.ndarray, shape (3,)
        The wavevector in lab coords, [k_z, k_y, k_x].
    reference_axis_lab : jnp.ndarray, shape (3,)
        Primary axis in [z, y, x] format, default [1,0,0].
    fallback_axes_lab : jnp.ndarray, shape (N,3)
        Fallback axes (each in [z, y, x]). If None, defaults to [[0,1,0],[0,0,1]].
    tol : float
        Tolerance for deciding near-zero norms.

    Returns
    -------
    M_lab_3x3 : jnp.ndarray, shape (3,3), complex128
        The embedded 3×3 matrix in [z, y, x] coordinate order.

    Note
    ----
    - If k_lab is near zero or no valid reference axis is found (k_hat parallel to 
      all axes), the function will return a degenerate basis (ŝ, p̂, etc. = 0).
      It's up to the caller to detect or handle that case.
    """
    if high_precision:
        dtype_float = jnp.float64
        dtype_complex = jnp.complex128
    else:
        dtype_float = jnp.float32
        dtype_complex = jnp.complex64

    # 0. Provide default fallback if none
    if fallback_axes_lab is None:
        fallback_axes_lab = jnp.array(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=dtype_float
        )

    # Stack reference_axis_lab on top of fallback_axes => shape (N+1, 3)
    ref_and_fallback = jnp.vstack([reference_axis_lab, fallback_axes_lab])  # float64

    # ------------------------
    # Utilities for coordinate reorder
    # ------------------------
    def zyx_to_xyz(vec_zyx: jnp.ndarray) -> jnp.ndarray:
        """Convert [z,y,x] -> [x,y,z]."""
        return jnp.array([vec_zyx[2], vec_zyx[1], vec_zyx[0]], dtype=dtype_float)

    def xyz_to_zyx(vec_xyz: jnp.ndarray) -> jnp.ndarray:
        """Convert [x,y,z] -> [z,y,x], returning complex128."""
        return jnp.array([vec_xyz[2], vec_xyz[1], vec_xyz[0]], dtype=dtype_complex)

    def xyz3x3_to_zyx3x3(M_xyz: jnp.ndarray) -> jnp.ndarray:
        """Reorder a (3×3) matrix from (x,y,z)->(z,y,x) in rows & cols."""
        perm = jnp.array([2, 1, 0])
        M_temp = M_xyz[perm, :]  # permute rows
        return M_temp[:, perm]   # permute columns

    # ------------------------
    # 1. Compute k_hat in (x,y,z)
    # ------------------------
    k_xyz = zyx_to_xyz(k_lab)  # float64[3]
    k_norm = jnp.linalg.norm(k_xyz)

    # If k_norm < tol => zero vector, else normalized
    def zero_khat():
        return jnp.zeros_like(k_xyz)

    def nonzero_khat():
        return k_xyz / k_norm

    k_hat = jax.lax.cond(k_norm < tol, zero_khat, nonzero_khat)  # float64[3]

    # ------------------------
    # 2. Choose s_hat among reference/fallback
    # ------------------------
    # Cross with each candidate
    fix_ordering = True
    if fix_ordering:
        # Convert reference_and_fallback from [z,y,x] -> [x,y,z]
        ref_and_fallback_xyz = jax.vmap(zyx_to_xyz)(ref_and_fallback)
        # Now do cross in [x,y,z]
        cross_vals = jnp.cross(k_hat, ref_and_fallback_xyz)
        # # Convert back to [z,y,x]
        # cross_vals = jax.vmap(zyx_to_xyz)(cross_vals)
    else:
        cross_vals = jnp.cross(k_hat, ref_and_fallback)  # shape (N+1,3), float64
    norms = jnp.linalg.norm(cross_vals, axis=1)
    valid = norms > tol
    first_valid_idx = jnp.argmax(valid)

    def has_valid_axis():
        return cross_vals[first_valid_idx] / norms[first_valid_idx]

    def no_valid_axis():
        return jnp.zeros(3, dtype=dtype_float)

    s_hat_xyz = jax.lax.cond(jnp.any(valid), has_valid_axis, no_valid_axis)  # float64[3]

    # ------------------------
    # 3. Compute p_hat = k_hat x s_hat
    # ------------------------
    p_xyz_unnorm = jnp.cross(k_hat, s_hat_xyz)
    p_norm = jnp.linalg.norm(p_xyz_unnorm)

    def has_nonzero_p():
        return p_xyz_unnorm / p_norm

    def zero_p():
        return jnp.zeros_like(p_xyz_unnorm)

    p_hat_xyz = jax.lax.cond(p_norm > tol, has_nonzero_p, zero_p)  # float64[3]

    if debug:
        print("s_hat =", s_hat_xyz)
        print("p_hat =", p_hat_xyz)
        print("k_hat =", k_hat)
        cross_sp = jnp.cross(s_hat_xyz, p_hat_xyz)
        print("cross(s_hat, p_hat) =", cross_sp, "should be parallel to k_hat")

        # Check orthogonality and sign
        print("dot(k_hat, s_hat) =", jnp.dot(k_hat, s_hat_xyz))
        print("dot(k_hat, p_hat) =", jnp.dot(k_hat, p_hat_xyz))
        print("dot(s_hat, p_hat) =", jnp.dot(s_hat_xyz, p_hat_xyz))

        # If you want to force s-hat first, p-hat second, you can also check:
        assert jnp.abs(jnp.dot(k_hat, s_hat_xyz)) < 1e-6, "k and s not orthogonal!"
        assert jnp.abs(jnp.dot(k_hat, p_hat_xyz)) < 1e-6, "k and p not orthogonal!"
        assert jnp.abs(jnp.dot(s_hat_xyz, p_hat_xyz)) < 1e-6, "s and p not orthogonal!"
        cross_sp_norm = jnp.linalg.norm(cross_sp)
        dot_with_k = jnp.dot(cross_sp / cross_sp_norm, k_hat)
        assert dot_with_k > 0.99, "s x p doesn't match k"


    # 4. Build W in (x,y,z): columns = [s_hat, p_hat, k_hat]
    # ------------------------
    W_xyz = jnp.column_stack([s_hat_xyz, p_hat_xyz, k_hat])  # shape (3,3), float64

    # ------------------------
    # 5. Embed M_wave_2x2 in 3×3
    # ------------------------
    # Build a 3×3 identity, then fill the top-left block with M_wave_2x2
    M_wave_2x2_cplx = M_wave_2x2.astype(dtype_complex)
    M_wave_3x3 = jnp.eye(3, dtype=dtype_complex)
    M_wave_3x3 = M_wave_3x3.at[0:2, 0:2].set(M_wave_2x2_cplx)

    # ------------------------
    # 6. Compute M_lab in (x,y,z): W_xyz * M_wave_3x3 * W_xyz^H
    # ------------------------
    W_xyz_cplx = W_xyz.astype(dtype_complex)  # (3,3), complex128
    # Hermitian transpose => .conj().T
    M_lab_xyz = W_xyz_cplx @ M_wave_3x3 @ W_xyz_cplx.conj().T  # (3,3), complex128

    # ------------------------
    # 7. Reorder from (x,y,z) -> (z,y,x)
    # ------------------------
    M_lab_3x3 = xyz3x3_to_zyx3x3(M_lab_xyz)  # (3,3), complex128

    return M_lab_3x3


def compute_angles_from_offset(x: float, y: float, focal_length: float) -> tuple[float, float]:
    """
    Compute the incidence angles (theta, phi) from the lateral offset (x, y)
    on the focal plane of a lens with focal_length.
    
    Args:
        x: x offset (in same units as focal_length)
        y: y offset (in same units as focal_length)
        focal_length: the focal length of the lens (used as z)
    
    Returns:
        A tuple (theta, phi) in radians, where:
          theta = arctan(r / focal_length)
          phi = arctan2(y, x)
    """
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan(r / focal_length)
    phi = jnp.arctan2(y, x)
    return theta, phi

# %%
