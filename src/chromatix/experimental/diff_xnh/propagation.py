import jax.numpy as jnp
from chromatix import ScalarField


def propagate(field: ScalarField, z: float, N_pad: int) -> ScalarField:
    """This does what kernel_propagate does, but pads using symmetric mode.
    Implemented to match holotomocupy."""
    N = field.shape[-3]
    padded = jnp.pad(field.u.squeeze(), N_pad, mode="symmetric")
    f = jnp.fft.fftfreq(N + 2 * N_pad, d=field.dx.squeeze()[0])
    f = jnp.stack(jnp.meshgrid(f, f), axis=0)
    kernel = jnp.exp(
        -1j * jnp.pi * field.spectrum.squeeze() * z * jnp.sum(f**2, axis=0)
    )
    propagated = jnp.fft.ifft2(jnp.fft.fft2(padded) * kernel)
    return field.replace(
        u=propagated[N_pad : N + N_pad, N_pad : N + N_pad][None, ..., None, None]
    )
