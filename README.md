# Chromatix 🔬: Differentiable wave optics using JAX!
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/chromatix-team/chromatix/blob/main/docs/media/logo_text_white.png?raw=true">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/chromatix-team/chromatix/blob/main/docs/media/logo_text_black.png?raw=true">
  <img alt="Chromatix logo" src="https://github.com/chromatix-team/chromatix/blob/main/docs/media/logo_text_black.png?raw=true">
</picture>

![CI](https://github.com/chromatix-team/chromatix/actions/workflows/test.yaml/badge.svg) ![Ruff](https://github.com/chromatix-team/chromatix/actions/workflows/format_lint.yaml/badge.svg)

[**Installation**](#installation)
| [**Usage**](#usage)
| [**Contributions**](#contributions)
| [**FAQ**](https://chromatix.readthedocs.io/en/latest/FAQ/)
| [**Chromatix Documentation**](https://chromatix.readthedocs.io/en/latest/)

Welcome to `chromatix`, a differentiable wave optics library built using `jax` which combines JIT-compilation, (multi-)GPU support, and automatic differentiation with a convenient programming style inspired by deep learning libraries. This makes `chromatix` a great fit for inverse problems in optics. We intend `chromatix` to be used by researchers in computational optics, so `chromatix` provides a set of optical element "building blocks" that can be composed together in a style similar to neural network layers. This means we take care of the more tedious details of writing fast optical simulations, while still leaving a lot of control over what is simulated and/or optimized up to you! Chromatix is still in active development, so **expect sharp edges**.

Here are some of the cool things we've already built with `chromatix`:

- [**Holoscope**](docs/examples/holoscope.ipynb): PSF engineering to optimally encode a 3D volume into a 2D image.
- [**Fourier Ptychography**](docs/examples/fourier_ptychography.ipynb): a simple demo of Fourier ptychography.
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
# install dependencies for development
$ pip install pytest ruff pre-commit
# install pre-commit hooks for formatting
pre-commit install
# test
$ pytest
```
Check out [the documentation](https://chromatix.readthedocs.io/en/latest/installing/) for more details on installation.

## Usage

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
When we obtain the intensity, if we had simulated in multiple wavelengths `chromatix` would take the spectrum and its density into account. This example uses only a single wavelength, but we can easily add more and `chromatix` will automatically adjust. We could also have checked the phase at the output instead: ``return field.phase`` and we would know the phase of the final PSF instead of the intensity.

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
* Cédric Allier ([@allierc](https://github.com/allierc))
* Changjia Cai ([@caichangjia](https://github.com/caichangjia))
* Geneva Schlafly ([@gschlafly](https://github.com/gschlafly))
* Guanghan Meng ([@guanghanmeng](https://github.com/guanghanmeng))
* Hoss Eybposh ([@hosseybposh](https://github.com/hosseybposh))
* Magdalena Schneider ([@schneidermc](https://github.com/schneidermc))
* Xi Yang ([@nicolexi](https://github.com/nicolexi))

and many more!

## Citation
To cite Chromatix, please refer to our 2025 preprint on bioRxiv:

**Deb, Diptodip**\* and **Both, Gert-Jan**\* and Bezzam, Eric and Kohli, Amit and Yang, Siqi and Chaware, Amey and Allier, Cédric and Cai, Changjia and Anderberg, Geneva and Eybposh, M. Hossein and Schneider, Magdalena C. and Heintzmann, Rainer and Rivera-Sanchez, Fabrizio A. and Simmerer, Corey and Meng, Guanghan and Tormes-Vaquerano, Jovan and Han, SeungYun and Shanmugavel, Sibi Chakravarthy and Maruvada, Teja and Yang, Xi and Kim, Yewon and Diederich, Benedict and Joo, Chulmin and Waller, Laura and Durr, Nicholas J. and Pégard, Nicolas C. and La Rivière, Patrick J. and Horstmeyer, Roarke and Chowdhury, Shwetadwip and Turaga, Srinivas C. *Chromatix*. bioRxiv. [https://doi.org/10.1101/2025.04.29.651152](https://doi.org/10.1101/2025.04.29.651152)

\* equal contribution

BibTex:
```bibtex
@article {Deb2025.04.29.651152,
	author = {Deb, Diptodip and Both, Gert-Jan and Bezzam, Eric and Kohli, Amit and Yang, Siqi and Chaware, Amey and Allier, C{\'e}dric and Cai, Changjia and Anderberg, Geneva and Eybposh, M. Hossein and Schneider, Magdalena C. and Heintzmann, Rainer and Rivera-Sanchez, Fabrizio A. and Simmerer, Corey and Meng, Guanghan and Tormes-Vaquerano, Jovan and Han, SeungYun and Shanmugavel, Sibi Chakravarthy and Maruvada, Teja and Yang, Xi and Kim, Yewon and Diederich, Benedict and Joo, Chulmin and Waller, Laura and Durr, Nicholas J. and Pegard, Nicolas C. and La Rivi{\`e}re, Patrick J. and Horstmeyer, Roarke and Chowdhury, Shwetadwip and Turaga, Srinivas C.},
	title = {Chromatix: a differentiable, GPU-accelerated wave-optics library},
	elocation-id = {2025.04.29.651152},
	year = {2025},
	doi = {10.1101/2025.04.29.651152},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/05/02/2025.04.29.651152},
	eprint = {https://www.biorxiv.org/content/early/2025/05/02/2025.04.29.651152.full.pdf},
	journal = {bioRxiv}
}
```

If you want to cite the repository specifically, you can use the following Zenodo citation:

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
