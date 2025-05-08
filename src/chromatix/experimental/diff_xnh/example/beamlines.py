import chromatix.functional as cf
import equinox as eqx
import numpy as np
from chromatix.field import crop
from jax import Array

from diff_xnh.probe import Probe, generic_field
from diff_xnh.propagation import propagate
from diff_xnh.sample import AbstractSample, thin_sample
from diff_xnh.shift import shift_field

kEv = float


class ProjectionConfig(eqx.Module):
    z_obj: Array
    angle: Array | None = None
    probe_shift: Array | None = None
    sample_shift: Array | None = None


class Beamline(eqx.Module):
    # Detector
    sensor_shape: tuple[int, int]
    pixel_pitch: float

    # Wavelength
    wavelength: float

    # distances
    z_total: float
    z_obj: Array

    # Padding
    N_pad_prop: int
    pad: int
    FoV: int

    def __init__(
        self,
        z_obj: Array,
        z_total: float,
        energy: kEv,
        sensor_shape: tuple[int, int],
        pixel_pitch: float,
        N_pad_prop: int | None = None,
        N_pad_sample: int | None = None,
    ):
        # Distances
        self.z_obj = z_obj
        self.z_total = z_total

        # wavelengths
        self.wavelength = 1.2398419840550367e-3 / energy

        # detector
        self.sensor_shape = sensor_shape
        self.pixel_pitch = pixel_pitch

        # Calculating sizes
        m_rel = self.z_ref / z_obj
        sample_size = sensor_shape[0]
        self.pad = sample_size // 16 if N_pad_sample is None else N_pad_sample
        FoV_float = (sample_size + 2 * self.pad) / m_rel[-1]
        self.FoV = int(np.ceil(FoV_float / 8) * 8)

        # Padding
        self.N_pad_prop = (
            (sample_size + 2 * self.pad) // 2 if N_pad_prop is None else N_pad_prop
        )

    def __call__(
        self, config: ProjectionConfig, sample: AbstractSample, probe: Probe
    ) -> Array:
        forward = eqx.filter_vmap(self.forward, in_axes=(0, None, None))
        return forward(config, sample, probe).u.squeeze()

    def forward(
        self, config: ProjectionConfig, sample: AbstractSample, probe: Probe
    ) -> cf.ScalarField:
        # Modified distances and magnifications
        m = self.z_total / config.z_obj
        m_rel = self.z_ref / config.z_obj

        # z_img_scaled
        z_img = self.z_total - config.z_obj
        z_img_scaled = z_img * m_rel**2 / m

        # z_obj_scaled
        z_obj_scaled = (config.z_obj - self.z_ref) * m_rel

        # Step 1a: making the probe field
        field = generic_field(
            dx=self.effective_resolution,
            spectrum=self.wavelength,
            spectral_density=1.0,
            amplitude=probe.amplitude,
            phase=probe.phase,
        )
        # Step 1b: shifting the probe
        if config.probe_shift is not None:
            field = shift_field(field, config.probe_shift)

        # Step 2: propagate probe field to right distance
        field = propagate(
            field,
            z=z_obj_scaled,
            N_pad=self.N_pad_prop,
        )

        # Step 3a: shift sample
        if config.sample_shift is not None:
            sample = sample.shift(config.sample_shift / m_rel)

        field = thin_sample(
            field,
            sample,
            config.angle,
            scale=m_rel * self.FoV / probe.amplitude.shape[1],
        )

        # Step 4: propagate field to detector
        field = propagate(field, z=z_img_scaled, N_pad=self.N_pad_prop)

        # Step 5: detector
        # We ignore the phase factor
        field = field.replace(_dx=field._dx * self.m_ref)
        return crop(field, self.pad)

    @property
    def z_ref(self):
        return self.z_obj[0]

    @property
    def m_ref(self):
        return self.z_total / self.z_ref

    @property
    def effective_resolution(self):
        return self.pixel_pitch / self.m_ref


def BinnedID16a() -> Beamline:
    z_obj = np.array([4.584e3, 4.765e3, 5.488e3, 6.9895e3]) - 3.7e2
    z_total = 1.28e6
    energy = 33.35  # kEV
    sensor_shape = (1024, 1024)
    pixel_pitch = 6.0  # normally 3.0, but data is binned

    return Beamline(z_obj, z_total, energy, sensor_shape, pixel_pitch)
