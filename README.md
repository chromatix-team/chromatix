# Chromatix 🔬: Differentiable wave optics using JAX!
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/chromatix-team/chromatix/blob/main/docs/media/logo_text_white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/chromatix-team/chromatix/blob/main/docs/media/logo_text_black.png?raw=true">
  <img alt="Chromatix logo" src="https://github.com/chromatix-team/chromatix/blob/main/docs/media/logo_text_black.png?raw=true">
</picture>

![CI](https://github.com/chromatix-team/chromatix/actions/workflows/test.yaml/badge.svg) ![Black](https://github.com/chromatix-team/chromatix/actions/workflows/black.yaml/badge.svg)

[**Installation**](#installation)
| [**Usage**](#usage)
| [**Contributions**](#contributions)
| [**FAQ**](https://chromatix.readthedocs.io/en/latest/FAQ/)
| [**Chromatix Documentation**](https://chromatix.readthedocs.io/en/latest/)

Welcome to `chromatix`, a differentiable wave optics library built using `jax` which combines JIT-compilation, (multi-)GPU support, and automatic differentiation with a convenient programming style inspired by deep learning libraries. This makes `chromatix` a great fit for inverse problems in optics. We intend `chromatix` to be used by researchers in computational optics, so `chromatix` provides a set of optical element "building blocks" that can be composed together in a style similar to neural network layers. This means we take care of the more tedious details of writing fast optical simulations, while still leaving a lot of control over what is simulated and/or optimized up to you! Chromatix is still in active development, so **expect sharp edges**.

Here are some of the cool things we've already built with `chromatix`:

- [**Holoscope**](docs/examples/holoscope.ipynb): PSF engineering to optimally encode a 3D volume into a 2D image.
- [**Computer Generated Holography**](docs/examples/cgh.ipynb): optimizing a phase mask to produce a 3D hologram.
- [**Aberration Phase Retrieval**](docs/examples/zernike_fitting.ipynb): fitting Zernike coefficients to a measured aberrated PSF.

## Installation

We recommend installing `jax` first as described in the [`jax` README](https://github.com/google/jax#pip-installation-gpu-cuda) in order to make sure that you install the version with appropriate CUDA support for running on GPUs, if desired.

Then, simply run
```bash
$ pip install git+https://github.com/chromatix-team/chromatix.git@main
```
or for an editable install for development, first clone the repository and then install as shown:
```bash
$ git clone https://github.com/chromatix-team/chromatix
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

Check out our full documentation at [https://chromatix.readthedocs.io/en/latest](https://chromatix.readthedocs.io/en/latest) for more details.

## Contributions

### New contributors

We're happy to take contributions of either examples, new optical elements, or expanded simulation capabilities (within reasonable scope)! Simply submit a pull request and we'll be happy to help you along. We're also grateful to people who find and report issues here, so we can fix or improve things as soon as possible.

### Contributor list
Chromatix was started by Diptodip Deb ([@diptodip](https://www.github.com/diptodip)), Gert-Jan Both ([@GJBoth](https://www.github.com/GJBoth)), and Srinivas C. Turaga ([@srinituraga](https://www.github.com/srinituraga)) at HHMI Janelia Research Campus, along with contributions by:

* Amey Chaware ([@isildur7](https://www.github.com/isildur7))
* Amit Kohli ([@apsk14](https://www.github.com/apsk14))
* Cédric Allier
* Changjia Cai ([@caichangjia](https://github.com/caichangjia))
* Geneva Schlafly ([@gschlafly](https://github.com/gschlafly))
* Guanghan Meng ([@guanghanmeng](https://github.com/guanghanmeng))
* Hoss Eybposh ([@hosseybposh](https://github.com/hosseybposh))
* Magdalena Schneider ([@schneidermc](https://github.com/schneidermc))
* Xi Yang ([@nicolexi](https://github.com/nicolexi))

## Citation
To cite this repository:

Deb, D.\*, Both, G.\*, Chaware, A., Kohli, A., Allier, C., Cai, C., Schlafly, G., Meng, G., Eybposh, M. H., Schneider, M., Yang, X., & Turaga, S. C. (2023). Chromatix. Zenodo. [https://doi.org/10.5281/zenodo.7803771](https://doi.org/10.5281/zenodo.7803771)

\* equal contribution

BibTex:
```bibtex
@software{chromatix_2023,
  author       = {Deb, Diptodip and
                  Both, Gert-Jan and
                  Chaware, Amey and
                  Kohli, Amit and
                  Allier, Cédric and
                  Cai, Changjia and
                  Schlafly, Geneva and
                  Meng, Guanghan and
                  Eybposh, M. Hossein and
                  Schneider, Magdalena and
                  Yang, Xi and
                  Turaga, Srinivas C.},
  title        = {Chromatix},
  month        = aug,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7803771},
  url          = {https://doi.org/10.5281/zenodo.7803771}
}
```

This citation entry represents the latest release of Chromatix. If you would like to cite a specific version, you can follow the DOI to Zenodo and choose a specific version there.
