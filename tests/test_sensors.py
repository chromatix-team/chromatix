import jax.numpy as jnp
from jax.random import PRNGKey
import chromatix.functional as cf
from chromatix.elements.sensors import BasicSensor


def test_basic_sensor():
    field = cf.objective_point_source(
        (512, 512), 0.3, 0.532, 1.0, jnp.linspace(-5, 5, num=3), f=100.0, n=1.0, NA=0.8
    )
    shape = (256, 256)
    spacing = 0.6
    key = PRNGKey(4)
    sensor = BasicSensor(
        shape, spacing, shot_noise_mode="poisson", resampling_method="cubic"
    )
    params = sensor.init({"params": key, "noise": key}, field)
    image = sensor.apply(params, field, rngs={"noise": key})
    assert image.shape[1:3] == shape
    assert image.shape[0] == field.shape[0]
    sensor = BasicSensor(
        shape,
        spacing,
        shot_noise_mode="poisson",
        resampling_method="pool",
        reduce_axis=0,
    )
    params = sensor.init({"params": key, "noise": key}, field)
    image = sensor.apply(params, field, rngs={"noise": key})
    assert image.squeeze().shape == shape
    assert image.shape[0] == 1
    params = sensor.init(
        {"params": key, "noise": key},
        field.intensity,
        input_spacing=field.dx[..., 0, 0].squeeze(),
    )
    image_from_intensity = sensor.apply(
        params,
        field.intensity,
        input_spacing=field.dx[..., 0, 0].squeeze(),
        rngs={"noise": key},
    )
    assert image_from_intensity.squeeze().shape == shape
    assert image_from_intensity.shape[0] == 1
    assert jnp.all(image_from_intensity == image)
