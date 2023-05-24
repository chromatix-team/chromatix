# Installing Chromatix

## System Requirements

Chromatix is based on [`jax`](https://github.com/google/jax) which only works
on macOS and Linux, not Windows.

If you would like to run simulations on GPU, you will need an NVIDIA GPU with
CUDA support.

!!! warning
    Installing `jax` automatically through dependencies in `pyproject.toml`
    can have some issues, as the CUDA version for your environment won't be
    automatically detected. We recommend installing `jax` first as described in
    the [`jax` README](https://github.com/google/jax#pip-installation-gpu-cuda)
    in order to make sure that you install the version with appropriate CUDA
    support for running on GPUs, if desired. Also see our section on installing
    with `conda` below if you wouuld like to avoid installing your own CUDA
    and/or building `jax` from source.

## Using `pip`

Once you have installed `jax`, you can install `chromatix` using:
```bash
$ pip install git+https://github.com/chromatix-team/chromatix.git
```
or for an editable install for development, first clone the repository and then install:
```bash
$ git clone https://github.com/chromatix-team/chromatix
$ cd chromatix
$ pip install -e .
```
Another option for development is to use a Python project management tool such
as [`Hatch`](https://hatch.pypa.io/latest/).

## Using `conda`

We do not package `chromatix` for `conda` because `jax` is also not officially
packaged for `conda`. However, if you would like to install `chromatix` into a `conda` environment
and also use a `conda` installation of CUDA, you can use the following instructions:

After creating and activating a `conda` environment with a supported Python version:
```bash
$ conda install -c conda-forge cudatoolkit=11.X
$ conda install -c conda-forge cudnn=A.B
$ conda install -c nvidia cuda-nvcc
$ pip install --upgrade "jax[cuda11_cudnnAB]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
$ pip install git+https://github.com/chromatix-team/chromatix.git
```
You will have to replace `X` above with the appropriate version supported by your graphics driver (e.g. `11.4`), and you must ensure
that `A` and `B` are the same for both the installation of `cudnn` and in the options when installing `jax` (e.g. `8.2` and `82`). You can see the versions of `cudnn`
for which `jax` has been packaged at: [https://storage.googleapis.com/jax-releases/jax_cuda_releases.html](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html).
