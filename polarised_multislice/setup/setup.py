from jax import Array
import chromatix.functional as cf
from ..tensor_tomo import thick_polarised_sample
from chromatix.ops import init_plane_resample

class Setup:
    # General
    n: float 

    # Initial field
    shape: tuple[int, int] = (1024, 1024)
    spacing: float
    wavelength: float
    init_angle: float

    # Universal compensator
    retA: float
    retB: float

    # 1st ff lens
    ff1_f: float
    ff1_NA: float

    # Sample background
    n_background: float
    sample_NA: float

    # 2nd ff lens
    ff2_f: float
    ff2_NA: float
    
    # MLA array
    mla_fs: Array
    mla_centers: Array
    mla_radii: Array

    # camera
    camera_shape: tuple[int, int]
    camera_pitch: float



def tensor_tomography(setup: Setup, potential, dz: float) -> Array:
    # Polarised plane wave as an input
    field = cf.plane_wave(shape=setup.shape, dx = setup.spacing, spectrum=setup.wavelength, spectral_density=1.0, scalar=False, amplitude=cf.linear(setup.init_angle))
    field = cf.universal_compensator(field, setup.retA, setup.retB)
    field = cf.ff_lens(field, setup.ff1_f, setup.n, setup.ff1_NA) 

    # We reverse propagate half a sample width, as the ff lens
    # is focused on the centre of the sample
    sample_size = potential.shape[0] * dz
    field = cf.transfer_propagate(field, -1/2 * sample_size, setup.n, N_pad=setup.shape[0], absorbing_boundary="super_gaussian")
    field = thick_polarised_sample(field, potential, setup.n_background, dz, setup.sample_NA)
    field = cf.transfer_propagate(field, -1/2 * sample_size, setup.n, N_pad=setup.shape[0], absorbing_boundary="super_gaussian")

    # ff lens, microarray and camera 
    # # Camera - should we add noise?
    field = cf.ff_lens(field, setup.ff2_f, setup.n, setup.ff2_NA) 
    field = cf.microlens_array(field, setup.n, setup.mla_fs, setup.mla_centers, setup.mla_radii, block_between=True)
    resample_fn = init_plane_resample(setup.camera_shape, setup.camera_pitch)
    return resample_fn(field.intensity.squeeze(), field.dx.squeeze())
