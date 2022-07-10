from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

with open('src/chromatix/_about.py') as f:
    exec(f.read(), globals())

setup(
    name="chromatix",
    version=__version__,
    description="Differentiable computational optics library using JAX!",
    long_description=long_description,
    url="https://github.com/TuragaLab/chromatix",
    author="Diptodip Deb",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="differentiable, simulation, computational optics, machine learning",
    packages=find_packages(exclude=["contrib", "docs", "tests"], where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "jax", "flax", "chex", "einops"],
    extras_require={
        "vis": ["matplotlib"],
        "dev": ["jupyter", "matplotlib"],
        "test": ["coverage"],
    },
)
