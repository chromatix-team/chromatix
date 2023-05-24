# Frequently asked questions

## How do I do use multiple GPUs?

Chromatix tries to respect composable `jax` transformations, so you can use all the tools `jax` gives you for working with multiple devices in whatever way is appropriate for your application.

* For explicitly controlling what chunks of an array are parallelized, have a look at [``pmap``](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap).
* For implicitly allowing `jax` to control parallelism, see [``Array``](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html), which allows you to program for multiple devices as if they were a single device.

We discuss these styles of parallelism in our documentation on [Parallelism](parallelism.md).

## How do I decide which parameters get optimized?
Any attribute of an element that is specified as a possibly trainable parameter can be initialized using `chromatix.utils.trainable` in order to make it trainable. Otherwise, the attribute will be initialized (using either an `Array`, `float`, or `Callable` that takes a shape argument as specified in the documentation for that function) as non-trainable state of that element. If you are initializing an attribute as trainable using an initialization function, then you can specify whether that function requires a `jax.random.PRNGKey` or not. For example, if you are initializing the pixels of a phase mask with the `flat_phase` function, then you can use `trainable(flat_phase, rng=False)` because `flat_phase` takes only a shape argument.

For example:

```python
import jax
from chromatix.elements import ThinLens, PhaseMask
from chromatix.utils import trainable, flat_phase

# Refractive index is trainable and initialized to 1.33
# Focal distance and NA are not trainable
model = ThinLens(10.0, trainable(1.33), 0.8)

# Focal distance is trainable and randomly initialized
# Refractive index and NA are not trainable
model = ThinLens(trainable(lambda key: 10.0 * jax.random.uniform(key)), 1.33, 0.8)

# Phase mask pixels are trainable and initialized with the right shape
# automatically at initialization time, and the pupil f, n, NA are not trainable
model = PhaseMask(trainable(flat_phase, rng=False), 1.33, 100.0, 0.8)
```

We discuss `trainable()` further in the [API documentation](api/utils.md#chromatix.utils.utils.trainable).

## How can I optimize parameters in `chromatix`?

Everything in Chromatix is just a PyTree or `jax`-transformable function, so you can use any `jax` optimization program or library you like! For deep learning style optimizers we suggest having a look at [optax](https://github.com/deepmind/optax). For classical optimization schemes, including optimizers found in `scipy` and conjugate gradient methods (but all implicitly differentiable!) see [`jaxopt`](https://github.com/google/jaxopt).
