# Installing Chromatix 

Chromatix is based on Jax which, in turn, does not work on windows. The instructions below thus only work on mac or linux. 

!!! warning
    Installing Jax through a pyproject.toml with cuda can have some issues, as it can't automatically detect your cuda version.
    If you're having issues with Jax, have a look at the [installation guide](https://github.com/google/jax#installation).
## Using poetry

We recommend using [Poetry](https://python-poetry.org/) as it automatically sets up a virtual environment for you.

```bash
git clone https://github.com/turagaLab/diffrax
cd chromatix
poetry install
```


## Using pip

Alternatively you can install Chromatix using pip. 

```bash
git clone https://github.com/turagaLab/diffrax
cd chromatix
pip install .
```

