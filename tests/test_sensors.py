import jax.numpy as jnp
from jax.random import PRNGKey

import chromatix.functional as cf
from chromatix.elements.sensors import BasicSensor


def test_basic_sensor():
    field = cf.objective_point_source(
        (512, 512),
        0.3,
        (0.532, 1.0),
        jnp.linspace(-5, 5, num=3),
        f=100.0,
        n=1.0,
        NA=0.8,
    )
    shape = (256, 256)
    spacing = 0.6
    key = PRNGKey(4)
    sensor = BasicSensor(
        shape, spacing, shot_noise_mode="poisson", resampling_method="cubic"
    )
    image = sensor(field, key=key)
    assert image.shape[1:] == shape
    assert image.shape[0] == field.shape[0]
    sensor = BasicSensor(
        shape,
        spacing,
        shot_noise_mode="poisson",
        resampling_method="pool",
        reduce_axis=0,
    )
    image = sensor(field, key=key)
    assert image.squeeze().shape == shape
    image_from_intensity = sensor(
        field.intensity, input_spacing=field.central_dx, key=key
    )
    assert image_from_intensity.squeeze().shape == shape
    assert jnp.all(image_from_intensity == image)
    sensor = BasicSensor(
        (512, 512),
        0.3,
        shot_noise_mode="poisson",
        resampling_method=None,
        reduce_axis=0,
    )
    no_resample_image = sensor(field, key=key)
    assert no_resample_image.squeeze().shape == field.spatial_shape
