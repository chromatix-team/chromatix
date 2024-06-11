from jax import Array
from numpy import ndarray

ArrayLike = Array | ndarray
ScalarLike = ArrayLike | float | int
ComplexScalarLike = ArrayLike | float | int | complex
