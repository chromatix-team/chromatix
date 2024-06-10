# Installing Chromatix

## System Requirements

Chromatix is based on [`jax`](https://github.com/google/jax) which can be
installed on macOS, Linux (Ubuntu), and Windows.

If you would like to run simulations on GPU, you will need an NVIDIA GPU with
CUDA support. Ubuntu installations can take advantage of NVIDIA GPUs assuming
a recent NVIDIA driver has been installed. Windows installations can take
advantage of NVIDIA GPUs by installing through WSL2 (Ubuntu) on an up to date
Windows 10+ installation with a recent NVIDIA driver installed on Windows.
Installations on macOS are CPU only on both Intel and Apple Silicon, with very
limited GPU support.

!!! warning
    Installing `jax` automatically through dependencies in `pyproject.toml` can
    have some issues, e.g. a CUDA capable version of `jax` will not be installed
    by default. We recommend installing `jax` first as described in the [`jax` README](https://github.com/google/jax?tab=readme-ov-file#installation)
    in order to make sure that you install the version with appropriate CUDA
    support for running on GPUs, if desired.

## Using `pip`

Chromatix can be installed on any supported operating system with Python 3.10+.
First install `jax` as described in the [`jax` README](https://github.com/google/jax?tab=readme-ov-file#installation).
NVIDIA support will be automatically installed if you install with `pip install jax["cuda12"]`.
Note that `jax` currently only supports CUDA 12. If your NVIDIA driver is compatible with CUDA 12
but is older than the version that the default `jax` installation is built for using `pip`, you
may see a warning when running your code that `jax` has disabled parallel compilation. This is
not an error and your code should still use the GPU, but it may take longer to compile before running.

!!! info
    If you are on Windows 10+ and want NVIDIA GPU support, first make sure
    you have an [up to date driver installed](https://www.nvidia.com/download/index.aspx)
    for Windows. Then, [install WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
    so that you have a terminal with Ubuntu running in WSL2. If you now install `jax`
    using the instructions above, you should automatically get GPU support.

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
Editable installations for development are recommended if you would like to
make changes to the internals of Chromatix or add new features (pull requests
welcomed!). Otherwise, please use the first installation command to get the
latest version of Chromatix.

Another option for development is to use a Python project management tool such
as [`Hatch`](https://hatch.pypa.io/latest/).

## Using `conda`

We do not package `chromatix` for `conda` because `jax` is also not officially
packaged for `conda`. However, if you would like to install `chromatix` into a
`conda` environment, you can [first create and activate a `conda` environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#)
with a supported Python version (3.10+), and then follow the `pip` installation instructions above.
