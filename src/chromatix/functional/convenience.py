import jax
import jax.numpy as jnp
import jax.lax as lax

from chromatix.field import Field, VectorField
from chromatix.typing import ScalarLike
from chromatix.utils import _squeeze_grid_to_2d
from chromatix.utils.fft import fft


def optical_fft(field: Field, z: ScalarLike, n: ScalarLike) -> Field:
    """
    Computes the optical ``fft`` or ``ifft`` on an incoming ``Field`` propagated
    by ``z``, depending on the sign of ``z`` (which is a scalar value that may
    be positive or negative). If ``z`` is positive an ``fft```will be performed,
    otherwise an ``ifft`` (due to the ``1 / (lambda * z)`` term in the single
    Fourier transform Fresnel propagation, which requires this behavior).
    The ``ifft`` is calculated in terms of the conjugate of the ``fft`` with
    appropriate normalization applied so that propagating forwards and then
    backwards yields the same ``Field`` up to numerical precision. This function
    also appropriately changes the sampling of the ``Field`` that is output
    (after propagating to some distance ``z``).
    Args:
        field: The ``Field`` to be propagated by ``fft``.
        z: The distance the ``Field`` will be propagated.
        n: Refractive index of the propagation medium.
    Returns:
        The propagated ``Field``, transformed by ``fft``/``ifft``.
    """
    L_sq = field.spectrum * z / n
    du = field.dk * jnp.abs(L_sq)
    # Forward transform normalization for z >= 0
    norm_fft = (z >= 0) * -1j * jnp.prod(field.dx, axis=0, keepdims=False) / L_sq
    # Inverse transform normalization for z < 0
    norm_ifft = (
        (z < 0)
        * -1j  # Sign change because we take the conjugate of the input
        * (L_sq / jnp.prod(du, axis=0, keepdims=False))  # Inverse length scale
        / jnp.prod(
            jnp.array(field.shape)  # Due to a different norm factor for fft and ifft
        )
    )
    # Inverse transform input needs to use the conjugate
    fft_input = (norm_fft * field.u) + (norm_ifft * field.conj.u)
    fft_output = fft(fft_input, axes=field.spatial_dims, shift=True)
    u = (z >= 0) * fft_output + (z < 0) * jnp.conj(fft_output)
    return field.replace(u=u, _dx=_squeeze_grid_to_2d(du, field.ndim))

def enforce_transversality(E_x, E_y, E_z, Theta, Phi):
    """
    Given a global field E_x, E_y, E_z at pupil angles (Theta, Phi),
    subtract out the longitudinal (k) component so that the result
    is purely transverse in the local spherical basis.

    Parameters
    ----------
    E_x, E_y, E_z : array-like
        Components of the electric field in global (x, y, z) coords.
        They should have the same shape as Theta, Phi.
    Theta, Phi : array-like
        Spherical angles (in radians) indexing each pupil point.
        shape must match E_x, E_y, E_z.

    Returns
    -------
    E_perp_x, E_perp_y, E_perp_z : array-like
        New field components in global coords, each with the same shape.
        These satisfy (E_perp . e_k) = 0 at each (Theta, Phi).
    """
    sin_th = jnp.sin(Theta)
    cos_th = jnp.cos(Theta)
    cos_ph = jnp.cos(Phi)
    sin_ph = jnp.sin(Phi)

    # Local direction of propagation: e_k
    e_kx = sin_th * cos_ph
    e_ky = sin_th * sin_ph
    e_kz = cos_th

    # Dot product E . e_k
    dot_Ek = E_x * e_kx + E_y * e_ky + E_z * e_kz

    # Subtract out longitudinal component
    E_perp_x = E_x - dot_Ek * e_kx
    E_perp_y = E_y - dot_Ek * e_ky
    E_perp_z = E_z - dot_Ek * e_kz

    return E_perp_x, E_perp_y, E_perp_z

def optical_debye_wolf(
    field,             # VectorField with shape (B, H, W, C, 3)
    z: float,          # focal length [µm]
    n: float,          # refractive index
    NA: float,         # numeric aperture
    nx_out: int = 128, # output-plane pixels in x
    ny_out: int = 128, # output-plane pixels in y
    x_range: float = 10.0, # size [µm] of output-plane region in x
    y_range: float = 10.0, # size [µm] of output-plane region in y
    transverse_bool: bool = False,
) -> VectorField:
    """
    Naive Debye–Wolf focusing into the focal plane z' = 0 for a lens of focal length `z`.
    All spatial distances (z, x_range, y_range, dx, dy, wavelength) are assumed to be in µm.

    Args:
        field: A VectorField with shape (B, H,  W, C, 3).
               We assume B=1, C=1 for simplicity; the last dimension is (Ez, Ey, Ex).
               All spatial sampling in `field.dx` and `field.spectrum[0]` must be in µm.
        z: Focal length in micrometers. If 0, a fallback of 1000.0 µm is used.
        n: Refractive index.
        NA: Numeric aperture => n * sin(alpha).
        nx_out, ny_out: Number of pixels in the focal-plane grid (x, y).
        x_range, y_range (µm): Physical size of the output-plane region in micrometers.

    Returns:
        A new VectorField of shape (1, ny_out, nx_out, 1, 3),
        containing (Ez, Ey, Ex) at the focal plane. 
        The new field.dx will reflect the sampling in the focal plane (in µm).
    """
    print(f"field power: {field.power.squeeze():.5e}")
    # -----------------------------------------------------------
    # 1) Parse & validate shapes
    # -----------------------------------------------------------
    b, h, w, c, pol_dim = field.u.shape
    assert b == 1, "For simplicity, only batch=1 is handled."
    assert c == 1, "For simplicity, only one wavelength (C=1) is handled."
    assert pol_dim == 3, "Last dimension must be (Ez, Ey, Ex)."

    if z == 0.0:
        z = 1000.0  # 1000 µm

    # Wavelength in µm (ensure it's a scalar float)
    lam_um = float(field.spectrum[0].squeeze())

    # Wave number (1/µm)
    k = (2.0 * jnp.pi * n) / lam_um

    # Half-cone angle
    alpha = jnp.arcsin(NA / n)
    lens_radius = z * jnp.tan(alpha)
    lens_diameter = 2 * lens_radius
    print(f"z: {z} µm, n: {n}, NA: {NA}, alpha: {alpha:.2f} radians => {alpha*180/jnp.pi:.2f} degrees, lens_diameter: {lens_diameter:.2f} µm")


    # -----------------------------------------------------------
    # 2) Pupil-plane dx, dy (in µm)
    # -----------------------------------------------------------
    dy_pupil = float(field.dx[0].squeeze())
    dx_pupil = float(field.dx[1].squeeze())

    # Coordinates in pupil plane
    y_vec = jnp.arange(h) - (h // 2)
    x_vec = jnp.arange(w) - (w // 2)

    Yp, Xp = jnp.meshgrid(y_vec * dy_pupil, x_vec * dx_pupil, indexing="xy")

    # Aperture radius
    Rp = jnp.sqrt(Xp**2 + Yp**2)
    pupil_mask = (Rp <= lens_radius).astype(field.u.dtype)

    # -----------------------------------------------------------
    # 3) Extract pupil-plane fields
    # -----------------------------------------------------------

    Ez_raw = field.u[0, :, :, 0, 0]    
    Ey_raw = field.u[0, :, :, 0, 1]
    Ex_raw = field.u[0, :, :, 0, 2]

    Theta = jnp.arctan2(Rp, z)
    Phi   = jnp.arctan2(Yp, Xp)

    if transverse_bool:
        Ex_perp, Ey_perp, Ez_perp = enforce_transversality(Ex_raw, Ey_raw, Ez_raw, Theta, Phi)
        print(f"Transversality enforced")
        # print the max and min of the components
        print(f"Ez_perp max: {jnp.max(Ez_perp)}, min: {jnp.min(Ez_perp)}")
        print(f"Ey_perp max: {jnp.max(Ey_perp)}, min: {jnp.min(Ey_perp)}")
        print(f"Ex_perp max: {jnp.max(Ex_perp)}, min: {jnp.min(Ex_perp)}")
    else:
        Ex_perp, Ey_perp, Ez_perp = Ex_raw, Ey_raw, Ez_raw

    # Print maximum components of all E fields in pupil plane
    print(f"Max pupil field components before masking:")
    print(f"  |Ex_pupil|: {jnp.max(jnp.abs(Ex_raw)):.3e}")
    print(f"  |Ey_pupil|: {jnp.max(jnp.abs(Ey_raw)):.3e}") 
    print(f"  |Ez_pupil|: {jnp.max(jnp.abs(Ez_raw)):.3e}")

    Ez_pupil = Ez_perp * pupil_mask
    Ey_pupil = Ey_perp * pupil_mask
    Ex_pupil = Ex_perp * pupil_mask

    # -----------------------------------------------------------
    # 4) Focal-plane coordinates
    # -----------------------------------------------------------
    x_out = jnp.linspace(-0.5 * x_range, 0.5 * x_range, nx_out)  # (nx_out,)
    y_out = jnp.linspace(-0.5 * y_range, 0.5 * y_range, ny_out)  # (ny_out,)
    Y_out, X_out = jnp.meshgrid(y_out, x_out, indexing="xy")

    # -----------------------------------------------------------
    # 5) Flatten pupil arrays
    # -----------------------------------------------------------
    Xp_flat  = Xp.ravel()
    Yp_flat  = Yp.ravel()
    Exp_flat = Ex_pupil.ravel()
    Eyp_flat = Ey_pupil.ravel()
    Ezp_flat = Ez_pupil.ravel()

    # Print maximum components of all E fields in pupil plane
    print(f"Max pupil field components after masking:")
    print(f"  |Ex_pupil|: {jnp.max(jnp.abs(Ex_pupil)):.3e}")
    print(f"  |Ey_pupil|: {jnp.max(jnp.abs(Ey_pupil)):.3e}") 
    print(f"  |Ez_pupil|: {jnp.max(jnp.abs(Ez_pupil)):.3e}")
    
    sin_t = jnp.sin(Theta)
    cos_t = jnp.cos(Theta)
    geom_factor = sin_t * jnp.sqrt(cos_t)
    geom_factor_flat = geom_factor.ravel()

    # Pupil area element in µm^2
    dA = dx_pupil * dy_pupil

    # Debye–Wolf prefactor
    prefac = (1j * k * jnp.exp(1j * k * z)) / (2.0 * jnp.pi * z)

    # -----------------------------------------------------------
    # 6) Contribution function (single pupil pixel -> single focal point)
    # -----------------------------------------------------------
    def contribution_per_pupil(idx, xo, yo):
        xp = Xp_flat[idx]
        yp = Yp_flat[idx]
        exP = Exp_flat[idx]
        eyP = Eyp_flat[idx]
        ezP = Ezp_flat[idx]
        gw  = geom_factor_flat[idx]

        rp = jnp.sqrt(xp**2 + yp**2)
        th = jnp.arctan2(rp, z)
        ph = jnp.arctan2(yp, xp)

        sin_th = jnp.sin(th)
        cos_th = jnp.cos(th)

        # Phase factor
        phase = k * (xo * sin_th * jnp.cos(ph) + yo * sin_th * jnp.sin(ph))
        E_phase = jnp.exp(1j * phase)

        # Spherical basis
        e_k     = jnp.array([sin_th*jnp.cos(ph), sin_th*jnp.sin(ph), cos_th])
        e_phi   = jnp.array([-jnp.sin(ph), jnp.cos(ph), 0.0])
        e_theta = jnp.array([cos_th*jnp.cos(ph), cos_th*jnp.sin(ph), -sin_th])

        # Pupil field in lab coords: (Ex, Ey, Ez)
        E_pupil = jnp.array([exP, eyP, ezP])
        # Project onto (e_k, e_theta, e_phi)
        E_k   = jnp.dot(E_pupil, e_k)
        E_th  = jnp.dot(E_pupil, e_theta)
        E_phi = jnp.dot(E_pupil, e_phi)

        E_foc = E_k * e_k + E_th * e_theta + E_phi * e_phi
        return E_foc * E_phase * gw

    # -----------------------------------------------------------
    # 7) Sum over pupil for each (xo, yo)
    # -----------------------------------------------------------
    def field_at_one_point(xo, yo):
        indices = jnp.arange(Xp_flat.size)
        E_xyz = jax.vmap(contribution_per_pupil, in_axes=(0, None, None))(indices, xo, yo)
        # E_xyz => shape (h*w, 3)

        E_sum = jnp.sum(E_xyz, axis=0) * dA  # shape (3,)

        # Reorder (Ex, Ey, Ez) -> (Ez, Ey, Ex) if needed
        Ex_val, Ey_val, Ez_val = E_sum
        return prefac * jnp.array([Ez_val, Ey_val, Ex_val])  # (3,)

    coords = jnp.stack([X_out.ravel(), Y_out.ravel()], axis=-1)  # (nx_out*ny_out, 2)

    def scan_fn(_, pt):
        return None, field_at_one_point(pt[0], pt[1])

    _, E_out_flat = jax.lax.scan(scan_fn, None, coords)
    # E_out_flat => (nx_out*ny_out, 3)

    # Reshape to (ny_out, nx_out, 3)
    E_out_2d = E_out_flat.reshape(ny_out, nx_out, 3)

    # -----------------------------------------------------------
    # 8) Build a new VectorField at the focal plane
    # -----------------------------------------------------------
    # Final shape => (1, ny_out, nx_out, 1, 3)
    E_out = E_out_2d[None, :, :, None, :]

    dx_out = x_range / nx_out
    dy_out = y_range / ny_out
    new_dx = jnp.array([[dy_out], [dx_out]], dtype=jnp.float32)

    p_in = field.power.squeeze()
    p_out = jnp.sum(jnp.abs(E_out)**2) * dx_out * dy_out
    print(f"Power conservation: {p_out / p_in:.5e}")

    E_out = E_out * jnp.sqrt(p_in / p_out)

    new_field = field.replace(
        u=E_out,
        _dx=new_dx,
    )
    print(f"New field power: {new_field.power.squeeze():.5e}")
    return new_field

def precompute_debye_geometry(
    Xp_flat, Yp_flat,
    Ex_flat, Ey_flat, Ez_flat,
    k, z, 
    dx_pupil, dy_pupil,
    pupil_mask_flat,
    geom_weight_flat
):
    """
    Precompute geometry & pupil transformations for each pupil point.

    Returns
    -------
    E0_pupil : complex array, shape (N_pupil, 3)
        The pupil's local contribution to the focal field at the *origin* (xo=0, yo=0).
        Already includes the lens factor e^{i k z}/z, the area element dA, 
        the geometric weighting, and the basis transformation from (Ex,Ey,Ez).
    kx, ky : float array, shape (N_pupil,)
        Wave-vector components for each pupil point in the focal plane.
    """

    Np = Xp_flat.shape[0]  # total number of pupil points

    # area element
    dA = dx_pupil * dy_pupil

    # Precompute spherical coords
    Rp = jnp.sqrt(Xp_flat**2 + Yp_flat**2)
    # angles
    Theta = jnp.arctan2(Rp, z)       # from 0 to alpha
    Phi   = jnp.arctan2(Yp_flat, Xp_flat)

    sin_th = jnp.sin(Theta)
    cos_th = jnp.cos(Theta)
    sin_ph = jnp.sin(Phi)
    cos_ph = jnp.cos(Phi)

    # Phase factor from lens to reference plane
    # This is constant for each pupil point if we only consider propagation along z.
    # But we incorporate it once in a prefac:
    prefac = (1j * k * jnp.exp(1j * k * z) / (2.0 * jnp.pi * z))

    # Geometric weight was given by: geom_weight = sin_th * sqrt(cos_th)
    # Multiply by area element:
    weight = geom_weight_flat * dA

    # -- Transform pupil field (Ex, Ey, Ez) to local spherical basis (e_k, e_th, e_phi)
    # We'll build those basis vectors in lab coords:
    # e_k = (sin_th*cos_ph, sin_th*sin_ph, cos_th)
    # e_phi = (-sin_ph, cos_ph, 0)
    # e_theta = (cos_th*cos_ph, cos_th*sin_ph, -sin_th)

    # Dot product E_pupil with e_k, e_theta, e_phi:
    def dot3(a, b):
        return jnp.sum(a*b, axis=-1)

    E_pupil = jnp.stack([Ex_flat, Ey_flat, Ez_flat], axis=-1)  # shape (Np, 3)

    e_k     = jnp.stack([sin_th*cos_ph, sin_th*sin_ph, cos_th], axis=-1)
    e_phi   = jnp.stack([-sin_ph,       cos_ph,        jnp.zeros(Np)], axis=-1)
    e_theta = jnp.stack([cos_th*cos_ph, cos_th*sin_ph, -sin_th], axis=-1)

    E_k     = jnp.sum(E_pupil * e_k,     axis=-1)
    E_ph    = jnp.sum(E_pupil * e_phi,   axis=-1)
    E_th    = jnp.sum(E_pupil * e_theta, axis=-1)

    # Reconstruct local field in lab coords at the focal reference (xo=0, yo=0)
    # E_foc = E_k e_k + E_th e_theta + E_ph e_phi
    # shape (Np, 3)
    E0_foc = (
        E_k[:, None]  * e_k
      + E_th[:, None] * e_theta
      + E_ph[:, None] * e_phi
    )

    # Now multiply by the combined scaling:
    # pupil_mask_flat is 0 or 1
    factor = prefac * weight * pupil_mask_flat
    E0_pupil = E0_foc * factor[:, None]

    # wavevector offsets for each pupil point:
    kx = k * sin_th * cos_ph
    ky = k * sin_th * sin_ph

    return E0_pupil, kx, ky

def optical_debye_wolf_factored_chunked(
    field,              # shape (1, H, W, 1, 3)
    z,                  # focal length
    n, 
    NA,
    nx_out=128,
    ny_out=128,
    x_range=10.0,
    y_range=10.0,
    transverse_bool=False,
    debug_bool=False,
    chunk_size=256
):
    """
    A Debye–Wolf calculation that factors out geometry and then
    chunks the focal-plane summation to avoid large memory usage.

    NOTE: we use JAX's lax.dynamic_slice_in_dim and dynamic_update_slice_in_dim
    to handle the chunked index slicing in a JIT-compatible way.
    The final output shape is (1, ny_out, nx_out, 1, 3), with the last axis
    in (Ez, Ey, Ex) order (as in your original code).
    """
    b, h, w, c, pol_dim = field.u.shape
    assert b == 1
    assert c == 1
    assert pol_dim == 3

    lam = field.spectrum[0].squeeze()  # [um]
    k = (2.0 * jnp.pi * n) / lam
    alpha = jnp.arcsin(NA / n)

    # sampling at the pupil
    dy_pupil = field.dx[0].squeeze()
    dx_pupil = field.dx[1].squeeze()

    # pupil-plane coordinates
    y_vec = jnp.arange(h) - (h // 2)
    x_vec = jnp.arange(w) - (w // 2)
    Yp, Xp = jnp.meshgrid(y_vec * dy_pupil, x_vec * dx_pupil, indexing="xy")

    # Extract the field (Ez, Ey, Ex) from last dimension
    Ez_pupil = field.u[0, :, :, 0, 0]
    Ey_pupil = field.u[0, :, :, 0, 1]
    Ex_pupil = field.u[0, :, :, 0, 2]

    # radial extent of pupil
    lens_radius = z * jnp.tan(alpha)
    pupil_mask = (Xp**2 + Yp**2 <= lens_radius**2).astype(Ez_pupil.dtype)

    # Optionally enforce transversality
    Theta = jnp.arctan2(jnp.sqrt(Xp**2 + Yp**2), z)
    Phi   = jnp.arctan2(Yp, Xp)
    if transverse_bool:
        Ex_perp, Ey_perp, Ez_perp = enforce_transversality(Ex_pupil, Ey_pupil, Ez_pupil, Theta, Phi)
    else:
        Ex_perp, Ey_perp, Ez_perp = Ex_pupil, Ey_pupil, Ez_pupil

    # Aperture
    Ex_pupil = Ex_perp * pupil_mask
    Ey_pupil = Ey_perp * pupil_mask
    Ez_pupil = Ez_perp * pupil_mask

    # Flatten pupil arrays
    Xp_flat  = Xp.ravel()
    Yp_flat  = Yp.ravel()
    Exp_flat = Ex_pupil.ravel()
    Eyp_flat = Ey_pupil.ravel()
    Ezp_flat = Ez_pupil.ravel()
    pupil_mask_flat = pupil_mask.ravel()

    # geometric weighting sin_th sqrt(cos_th)
    Rp = jnp.sqrt(Xp**2 + Yp**2).ravel()
    Theta_flat = jnp.arctan2(Rp, z)
    sin_th = jnp.sin(Theta_flat)
    cos_th = jnp.cos(Theta_flat)
    geom_weight_flat = sin_th * jnp.sqrt(cos_th)

    # Precompute => E0_pupil: (Npupil, 3), plus kx, ky => (Npupil,)
    E0_pupil, kx, ky = precompute_debye_geometry(
        Xp_flat, Yp_flat,
        Exp_flat, Eyp_flat, Ezp_flat,
        k, z,
        dx_pupil, dy_pupil,
        pupil_mask_flat, geom_weight_flat
    )
    # E0_pupil => shape (Npupil, 3)

    # Build output-plane coords => (nx_out*ny_out,)
    x_out = jnp.linspace(-0.5 * x_range, 0.5 * x_range, nx_out)
    y_out = jnp.linspace(-0.5 * y_range, 0.5 * y_range, ny_out)
    Y_out, X_out = jnp.meshgrid(y_out, x_out, indexing="xy")
    X_out_flat = X_out.ravel()  # (NxNy,)
    Y_out_flat = Y_out.ravel()  # (NxNy,)

    NxNy = X_out_flat.size  # total # of focal-plane points

    assert NxNy % chunk_size == 0
    num_chunks = NxNy // chunk_size

    # Prepare storage for the flattened output => (NxNy, 3)
    dtype = jnp.complex128 if field.u.dtype == jnp.complex128 else jnp.complex64
    E_out_flat_init = jnp.zeros((NxNy, 3), dtype=dtype)

    # We'll define a helper that processes exactly 'chunk_size' each iteration.
    def process_chunk(i, E_accum):
        start = i * chunk_size
        # 'size' is now just chunk_size, a Python integer => fully static
        x_chunk = lax.dynamic_slice_in_dim(X_out_flat, start, chunk_size)
        y_chunk = lax.dynamic_slice_in_dim(Y_out_flat, start, chunk_size)

        phase = jnp.exp(1j * (kx[:, None] * x_chunk[None, :]
                            + ky[:, None] * y_chunk[None, :]))
        # => shape (Npupil, chunk_size)
        E_chunk = jnp.dot(E0_pupil.T, phase).T  # => (chunk_size, 3)
        E_accum = lax.dynamic_update_slice_in_dim(E_accum, E_chunk, start, axis=0)
        return (i+1, E_accum)

    # Now we unroll i=0..(num_chunks-1) with a lax.fori_loop or a lax.scan
    # Option A: lax.fori_loop
    def body_fun(i, carry_in):
        # i is a *static* integer from 0..num_chunks, so no tracer trouble
        _, E_accum_out = process_chunk(i, carry_in)
        return E_accum_out

    E_out_flat = lax.fori_loop(0, num_chunks, body_fun, E_out_flat_init)

    # Reshape => (ny_out, nx_out, 3)
    E_out_2d = E_out_flat.reshape(ny_out, nx_out, 3)

    # Right now, E_out_2d has (Ex, Ey, Ez) in that last axis from the dot-product approach.
    # If you want (Ez, Ey, Ex), reorder here:
    E_out_2d = E_out_2d[..., [2,1,0]]

    # Add batch/channel dims => (1, ny_out, nx_out, 1, 3)
    E_out = E_out_2d[None, :, :, None, :]

    # output dx, dy
    dx_out = x_range / nx_out
    dy_out = y_range / ny_out
    new_dx = jnp.array([[dy_out], [dx_out]], dtype=jnp.float32)

    # Power normalization
    p_in = field.power.squeeze()
    p_out = jnp.sum(jnp.abs(E_out)**2) * dx_out * dy_out
    E_out = E_out * jnp.sqrt(p_in / p_out)

    # Construct new field
    new_field = field.replace(u=E_out, _dx=new_dx)

    return new_field
