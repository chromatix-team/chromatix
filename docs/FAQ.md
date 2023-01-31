# Frequently asked questions

## How do I do use multiple GPUs?

Chromatix is just a thin layer around Jax, so you can use all the tools it gives you for working with multiple devices. 

* For data-paralleism have a look at the documentation for [`pmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap).
* For model-parallelism use [`Array`](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html), which allows you to use a node of GPUs as if it's a single device.

We also discuss some strategies for parallelization in Chromatix 101.

## How do I decide which parameters get optimized?
We use simple pattern to decide which parameters are optimizable: to make a parameter trainable, it should be a function with an rng key as input and the initial value as output.

A few examples:

```py
# Focal distance is trainable, random init
model = ce.ThinLens(lambda key: 10 * random.uniform(key), 1.33, 0.8)

# Refractive index is trainable, fixed init 
model = ce.ThinLens(10.0, lambda _: 1.33, 0.8)
```

## How can I use conjugate gradient?

Everything in Chromatix is just a pytree or jax-transformable function, so you can use any package you like! For deep-learning style optimizers we suggest having a look at [optax](https://github.com/deepmind/optax). For classical optimization schemes, including scipy optimizers and conjugate gradient (but all implicitly differentiable!) we use [jaxopt](https://github.com/google/jaxopt). 

