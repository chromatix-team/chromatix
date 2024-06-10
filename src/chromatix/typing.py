from jax import Array
from numpy import ndarray

ArrayLike = Array | ndarray
NumberLike = ArrayLike | float | int
ComplexNumberLike = ArrayLike | float | int | complex
