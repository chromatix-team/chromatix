from __future__ import annotations

from typing import Any, Callable, Sequence

import equinox as eqx
from jax import Array

from chromatix import Field


class OpticalSystem(eqx.Module):
    """
    Combines a sequence of optical elements into a single ``Module``.

    Takes a sequence of functions or ``Module``s (any ``Callable``) and calls
    them in sequence, assuming each element of the sequence only accepts a
    ``Field`` as input and returns a ``Field`` as output, with the exception of
    the first element of the sequence, which can take any arguments necessary
    (e.g. to allow an element from ``chromatix.elements.sources`` to initialize
    a ``Field``) and the last element of the sequence, which may return an
    ``Array``. This is intended to mirror the style of deep learning libraries
    that describe a neural network as a sequence of layers, allowing for an
    optical system to be described conveniently as a list of elements.

    Attributes:
        elements: A sequence of optical elements describing the system.
    """

    elements: Sequence[Callable]

    def __init__(self, elements: Sequence[Callable]):
        self.elements = elements

    def __call__(self, *args: Any, **kwargs: Any) -> Field | Array:
        """Returns the result of calling all elements in sequence."""
        # NOTE(dd/2025-08-12): Allow the first element to be a source
        # generating a Field, which would require additional arguments
        field = self.elements[0](*args, **kwargs)
        for element in self.elements[1:]:
            field = element(field)
        return field
