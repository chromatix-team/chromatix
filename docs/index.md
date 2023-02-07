# Chromatix: Differentiable lightfield simulation 

Chromatix is a lightfield simulator based on [Jax](https://github.com/google/jax). Because of this we're fast, support multi-GPU setups and are fully differentiable. 
It's first and foremost written for *research*: we offer a set of commonly required building blocks for working with lightfields, and leave the rest to you.
This gives you the freedom to build whatever you can come up without us getting in the way. A few of the things we've build with it:

- [**Holoscope**](examples/holoscope.ipynb) - optimizing the phasemask of an SLM to optimally encode a 3D volume into a 2D image. 
- [**Fourier Ptychograpy**](examples/fourier_ptychography.md) - parallel fourier ptychography.
- [**Synchotron X-ray tomography**](examples/tomography.md) - large scale phase constrast imaging with learnable parameters.

Have a look [here](installing.md) for how to install Chromatix, and read [Chromatix 101](101.md) to get started. 