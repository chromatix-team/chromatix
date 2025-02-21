# Chromatix 🔬: Differentiable wave optics using JAX!

!!! tip "Hackathon" 
    We're organising a Chromatix Hackathon at Janelia Research Campus May 27th - 31st. See [here](hackathon.md) for more info and how to apply!


Welcome to `chromatix`, a differentiable wave optics library built using `jax` which combines JIT-compilation, (multi-)GPU support, and automatic differentiation with a convenient programming style inspired by deep learning libraries. This makes `chromatix` a great fit for inverse problems in optics. We intend `chromatix` to be used by researchers in computational optics, so `chromatix` provides a set of optical element "building blocks" that can be composed together in a style similar to neural network layers. This means we take care of the more tedious details of writing fast optical simulations, while still leaving a lot of control over what is simulated and/or optimized up to you! Chromatix is still in active development, so **expect sharp edges**.

Here are some of the cool things we've already built with `chromatix`:

- [**Holoscope**](examples/holoscope.ipynb): PSF engineering to optimally encode a 3D volume into a 2D image.
- [**Computer Generated Holography**](examples/cgh.ipynb): optimizing a phase mask to produce a 3D hologram.
- [**Aberration Phase Retrieval**](examples/zernike_fitting.ipynb): fitting Zernike coefficients to a measured aberrated PSF.

Chromatix describes optical systems as sequences of sources and optical elements, composed in a similar style as neural network layers. These elements pass `Field` objects to each other, which contain both the tensor representation of the field at particular planes as well as information about the spatial sampling of the field and its spectrum. Typically, a user will not have to construct or deal with these `Field` objects unless they want to, but they are how `chromatix` can keep track of a lot of details of a simulation under the hood. Here's a very brief example of using `chromatix` to calculate the intensity of a widefield PSF (point spread function) at a single wavelength by describing a 4f system with a flat phase mask:

```python
import chromatix
import chromatix.elements
import jax
import jax.numpy as jnp
shape = (512, 512) # number of pixels in simulated field
spacing = 0.3 # spacing of pixels for the final PSF, microns
spectrum = 0.532 # microns
spectral_density = 1.0
f = 100.0 # focal length, microns
n = 1.33 # refractive index of medium
NA = 0.8 # numerical aperture of objective
optical_model = chromatix.OpticalSystem(
    [
        chromatix.elements.ObjectivePointSource(shape, spacing, spectrum, spectral_density, f, n, NA),
        chromatix.elements.PhaseMask(jnp.ones(shape)),
        chromatix.elements.FFLens(f, n)
    ]
)
# Calculate widefield PSF at multiple defocuses in parallel.
# We first have to initialize any parameters or state of the system:
variables = optical_model.init(jax.random.PRNGKey(4), jnp.linspace(-5, 5, num=11))
widefield_psf = optical_model.apply(variables, jnp.linspace(-5, 5, num=11)).intensity
```
When we obtain the intensity, `chromatix` took the spectrum as described by `spectrum` and `spectral_density` into account. This example uses only a single wavelength, but we can easily add more and `chromatix` will automatically adjust. We could also have checked the spacing at the output: ``optical_model.apply(variables, jnp.linspace(-5, 5, num=11)).dx`` and we would know the pixel spacing of the final PSF.

Chromatix supports a variety of optical phenomena and elements including:

* phase masks
* amplitude masks
* lenses
* wave propagation
* multiple wavelengths
* polarization
* shot noise simulation and sensors

Once you've [installed](https://chromatix.readthedocs.io/en/latest/installing/) `chromatix`, have a look through our [getting started guide](https://chromatix.readthedocs.io/en/latest/101/)!
