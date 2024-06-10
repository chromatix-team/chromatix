from __future__ import annotations

from typing import Any, Callable, Sequence, Union

from chex import Array
from flax import linen as nn

from ..field import Field


class OpticalSystem(nn.Module):
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

    @nn.compact
    def __call__(self, *args: Any, **kwargs: Any) -> Union[Field, Array]:
        """Returns the result of calling all elements in sequence."""
        field = self.elements[0](*args, **kwargs)  # allow field to be initialized
        for element in self.elements[1:]:
            field = element(field)
        return field
