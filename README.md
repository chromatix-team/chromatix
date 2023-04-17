# Chromatix 🔬: Differentiable wave optics library using JAX!

![CI](https://github.com/TuragaLab/chromatix/actions/workflows/test.yaml/badge.svg) ![Black](https://github.com/TuragaLab/chromatix/actions/workflows/black.yaml/badge.svg)

[**Installation**](#installation)
| [**Usage**](#usage)
| [**Contributions**](#contributions)
| [**FAQ**](https://chromatix.readthedocs.io/en/latest/FAQ/)
| [**Chromatix Documentation**](https://chromatix.readthedocs.io/en/latest/)

Welcome to `chromatix`, a differentiable wave optics library built using `jax` which combines JIT-compilation, (multi-)GPU support, and automatic differentiation with a convenient programming style inspired by deep learning libraries. This makes `chromatix` a great fit for inverse problems in optics. We intend `chromatix` to be used by researchers in computational optics, so `chromatix` provides a set of optical element "building blocks" that can be composed together in a style similar to neural network layers. This means we take care of the more tedious details of writing fast optical simulations, while still leaving a lot of control over what is simulated and/or optimized up to you! Chromatix is still in active development, so **expect sharp edges**.

Here are some of the cool things we've already built with `chromatix`:

- [**Holoscope**](docs/examples/holoscope.ipynb): optimizing a phase mask to optimally encode a 3D volume into a 2D image. 
- [**Fourier Ptychograpy**](docs/examples/fourier_ptychography.md): differentiable simulation of Fourier ptychography.
- [**Synchrotron X-ray Tomography**](docs/examples/tomography.md): large scale phase constrast imaging with learnable parameters.

## Installation

We recommend installing `jax` first as described in the [`jax` README](https://github.com/google/jax#pip-installation-gpu-cuda) in order to make sure that you install the version with appropriate CUDA support for running on GPUs, if desired.

Then, simply run
```bash
$ pip install git+https://github.com/TuragaLab/chromatix.git@main
```
or for an editable install for development, first clone the repository and then install as shown:
```bash
$ git clone https://github.com/TuragaLab/chromatix
$ cd chromatix
$ pip install -e .
```
Check out [the documentation](https://chromatix.readthedocs.io/en/latest/installing/) for more details on installation.

## Usage

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
# This system has no optimizable parameters, so we have to
# pass an empty parameter dictionary when calling the system:
widefield_psf = optical_model.apply({}, jnp.linspace(-5, 5, num=11)).intensity
```
When we obtain the intensity, `chromatix` took the spectrum as described by `spectrum` and `spectral_density` into account. This example uses only a single wavelength, but we can easily add more and `chromatix` will automatically adjust. We could also have checked the spacing at the output: ``optical_model.apply({}, jnp.linspace(-5, 5, num=11)).dx`` and we would know the pixel spacing of the final PSF.

Chromatix supports a variety of optical phenomena and elements including:

* phase masks
* amplitude masks
* lenses
* wave propagation
* multiple wavelengths
* polarization
* shot noise simulation and sensors

Check out our full documentation at [https://chromatix.readthedocs.io/en/latest](https://chromatix.readthedocs.io/en/latest) for more details.

## Contributions

### New contributors

We're happy to take contributions of either examples, new optical elements, or expanded simulation capabilities (within reasonable scope)! Simply submit a pull request and we'll be happy to help you along. We're also grateful to people who find and report issues here, so we can fix or improve things as soon as possible.

### Contributor list
Chromatix was started by Diptodip Deb, Gert-Jan Both, and Srinivas C. Turaga at HHMI Janelia Research Campus, along with contributions by:

* Amey Chaware
* Amit Kholi
* Cédric Allier
* Changjia Cai
* Geneva Schlafly
* Guanghan Meng
* Hoss Eybposh
* Magdalena Schneider
* Xi Yang
 
