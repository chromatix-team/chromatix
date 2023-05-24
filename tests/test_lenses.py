import chromatix.functional as cf
import jax.numpy as jnp


def test_ff_lens():
    field_after_first_lens = cf.objective_point_source(
        (512, 512), 0.3, 0.532, 1.0, 0, f=10.0, n=1.0, NA=0.8
    )
    field_after_second_lens = cf.ff_lens(field_after_first_lens, f=10.0, n=1, NA=None)
    field_after_third_lens = cf.ff_lens(field_after_second_lens, f=10.0, n=1, NA=None)
    field_after_second_lens_back = cf.ff_lens(
        field_after_third_lens, f=10.0, n=1, NA=None, inverse=True
    )

    assert jnp.allclose(
        field_after_second_lens.intensity,
        field_after_second_lens_back.intensity,
        atol=1e-5,
    )

    assert field_after_third_lens.u.squeeze()[256, 256] != 0.0


def test_df_lens():
    field_after_first_lens = cf.objective_point_source(
        (512, 512), 0.3, 0.532, 1.0, 0, f=10.0, n=1.0, NA=0.8
    )
    field_after_second_lens = cf.df_lens(
        field_after_first_lens, d=8.0, f=10.0, n=1, NA=None
    )
    field_after_third_lens = cf.df_lens(
        field_after_second_lens, d=8.0, f=10.0, n=1, NA=None
    )
    field_after_second_lens_back = cf.df_lens(
        field_after_third_lens, d=8.0, f=10.0, n=1, NA=None, inverse=True
    )

    # We don't test the exact fields as their spacing is different
    assert jnp.allclose(
        field_after_second_lens.power, field_after_second_lens_back.power
    )

    assert field_after_third_lens.u.squeeze()[256, 256] != 0.0
