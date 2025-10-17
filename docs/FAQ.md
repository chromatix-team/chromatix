# Frequently asked questions

## How do I do use multiple GPUs?

Chromatix tries to respect composable `jax` transformations, so you can use all the tools `jax` gives you for working with multiple devices in whatever way is appropriate for your application.

* For explicitly controlling what chunks of an array are parallelized, have a look at [``pmap``](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap).
* For implicitly allowing `jax` to control parallelism, see [``Array``](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html), which allows you to program for multiple devices as if they were a single device.

We discuss these styles of parallelism in our documentation on [Parallelism](parallelism.md).

## How do I decide which parameters get optimized?
You can either create a function or Equinox/Flax `Module` accepting only your optimizable parameters and use those parameters in the implementation *or* use an ``OpticalSystem`` and Equinox's ``partition`` and ``combine`` functions to select the attributes of the model that you would like to optimize. For examples of how to do this for all three approaches, see our [documentation on optimization](https://chromatix.readthedocs.io/en/latest/training/).


## How can I optimize parameters in `chromatix`?

Everything in Chromatix is just a PyTree or `jax`-transformable function, so you can use any `jax` optimization program or library you like! For deep learning style optimizers we suggest having a look at [optax](https://github.com/deepmind/optax). For classical optimization schemes, including optimizers found in `scipy` and conjugate gradient methods (but all implicitly differentiable!) see [`jaxopt`](https://github.com/google/jaxopt). Also see our [documentation on optimization](https://chromatix.readthedocs.io/en/latest/training/).
