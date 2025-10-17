# Chromatix 🔬: Differentiable wave optics using JAX!

Welcome to `chromatix`, a differentiable wave optics library built using `jax` which combines JIT-compilation, (multi-)GPU support, and automatic differentiation with a convenient programming style inspired by deep learning libraries. This makes `chromatix` a great fit for inverse problems in optics. We intend `chromatix` to be used by researchers in computational optics, so `chromatix` provides a set of optical element "building blocks" that can be composed together in a style similar to neural network layers. This means we take care of the more tedious details of writing fast optical simulations, while still leaving a lot of control over what is simulated and/or optimized up to you! Chromatix is still in active development, so **expect sharp edges**.

Here are some of the cool things we've already built with `chromatix`:

- [**Holoscope**](examples/holoscope.ipynb): PSF engineering to optimally encode a 3D volume into a 2D image.
- [**Fourier Ptychography**](examples/fourier_ptychography.ipynb): a simple demo of Fourier ptychography.
- [**Computer Generated Holography**](examples/cgh.ipynb): optimizing a phase mask to produce a 3D hologram.
- [**Aberration Phase Retrieval**](examples/zernike_fitting.ipynb): fitting Zernike coefficients to a measured aberrated PSF.

Chromatix describes optical systems as sequences of sources and optical elements, composed in a style similar to neural network layers. These elements pass `Field` objects to each other, which contain both the tensor representation of the field at particular planes as well as information about the spatial sampling of the field and its spectrum. Typically, a user will not have to construct or deal with these `Field` objects unless they want to, but they are how `chromatix` can keep track of a lot of details of a simulation under the hood. Here's a very brief example of using `chromatix` to calculate the intensity of a widefield PSF (point spread function) at a single wavelength by describing a 4f system with a flat phase mask:

```python
import chromatix.functional as cx
import jax
import jax.numpy as jnp
shape = (1280, 1280) # number of pixels in simulated field
camera_pixel_pitch = 6.5 # spacing of pixels for the final PSF at the camera, microns
spectrum = 0.532 # microns
f_objective = 8e3 # focal length, microns
f_tube = 200.0e3
n = 1.33 # refractive index of medium
NA = 0.6 # numerical aperture of objective
spacing = f_tube * spectrum / (n * shape[0] * camera_pixel_pitch) # spacing for simulation


@jax.jit
def optical_model(z: jax.Array) -> jax.Array:
    # Field in the Fourier plane due to a point source defocused by z from the
    # focal plane through an objective
    field = cx.objective_point_source(shape, spacing, spectrum, z, f_objective, n, NA)
    #  Flat phase mask in the Fourier plane
    field = cx.phase_change(field, jnp.ones(shape))
    # Field at the image plane after the tube lens
    field = cx.ff_lens(field, f_tube, n)
    # Return intensity of field
    return field.intensity


# Calculate widefield PSF at multiple defocuses in parallel.
# We first have to initialize any parameters or state of the system:
widefield_psf = optical_model(jnp.linspace(-5, 5, num=11))
```
When we obtain the intensity, `chromatix` took the spectrum as described by `spectrum` and `spectral_density` into account. This example uses only a single wavelength, but we can easily add more and `chromatix` will automatically adjust. We could also have checked the phase at the output instead: ``return field.phase`` and we would know the phase of the final PSF instead of the intensity.

Chromatix supports a variety of optical phenomena and elements including:

* phase masks
* amplitude masks
* lenses
* wave propagation
* multiple wavelengths
* polarization
* shot noise simulation and sensors

Once you've [installed](https://chromatix.readthedocs.io/en/latest/installing/) `chromatix`, have a look through our [getting started guide](https://chromatix.readthedocs.io/en/latest/101/)!
