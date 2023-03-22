import jax.numpy as jnp
from jax.random import PRNGKey
from chromatix import Field
import chromatix.functional as cf
from chromatix.elements.sensors import ShotNoiseIntensitySensor


def test_shot_noise_intensity_sensor():
    field = Field.create(0.3, 0.532, 1.0, shape=(512, 512))
    field = cf.objective_point_source(
        field, jnp.linspace(-5, 5, num=3), f=100.0, n=1.0, NA=0.8
    )
    shape = (256, 256)
    spacing = 0.6
    key = PRNGKey(4)
    sensor = ShotNoiseIntensitySensor(
        shape, spacing, shot_noise_mode="poisson", resampling_method="cubic"
    )
    params = sensor.init({"params": key, "noise": key}, field)
    image = sensor.apply(params, field, rngs={"noise": key})
    assert image.shape[1:3] == shape
    assert image.shape[0] == field.shape[0]
    sensor = ShotNoiseIntensitySensor(
        shape,
        spacing,
        shot_noise_mode="poisson",
        resampling_method="pool",
        reduce_axis=0,
    )
    params = sensor.init({"params": key, "noise": key}, field)
    image = sensor.apply(params, field, rngs={"noise": key})
    assert image.shape[1:3] == shape
    assert image.shape[0] == 1
