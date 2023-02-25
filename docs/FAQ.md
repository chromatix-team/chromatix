# Frequently asked questions

## How do I do use multiple GPUs?

Chromatix tries to respect composable `jax` transformations, so you can use all the tools `jax` gives you for working with multiple devices in whatever way is appropriate for your application.

* For explicitly controlling what chunks of an array are parallelized, have a look at [``pmap``](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap).
* For implicitly allowing `jax` to control parallelism, see [``Array``](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html), which allows you to program for multiple devices as if they were a single device.

We discuss these styles of parallelism in our documentation on [Parallelism](parallelism.md).

## How do I decide which parameters get optimized?
We use a simple pattern (following `flax.linen.Module` conventions) to decide which parameters are optimizable: to make a parameter trainable, it should be a function taking a ``jax.random.PRNGKey`` as input and outputting whatever initial value is desired.
We provide a utility function called `trainable()` in order to make specifying trainable parameters and their initial values convenient. At the same time, arbitrary functions can be passed as intiailizers as long as they have the appropriate function signature. This allows for a lot of flexibility in how parameters are intiailized.

For example:

```python
from chromatix.elements import ThinLens
from chromatix.utils import trainable

# Refractive index is trainable and initialized to 1.33
# Focal distance is not trainable
model = ThinLens(10.0, trainable(1.33), 0.8)

# Focal distance is trainable and randomly initialized
# Refractive index is not trainable
model = ThinLens(lambda key: 10 * random.uniform(key), 1.33, 0.8)
```

We discuss `trainable()` further in the [API documentation](api/utils.md#chromatix.utils.utils.trainable).

## How can I optimize parameters in `chromatix`?

Everything in Chromatix is just a PyTree or `jax`-transformable function, so you can use any `jax` optimization program or library you like! For deep learning style optimizers we suggest having a look at [optax](https://github.com/deepmind/optax). For classical optimization schemes, including optimizers found in `scipy` and conjugate gradient methods (but all implicitly differentiable!) see [`jaxopt`](https://github.com/google/jaxopt).
