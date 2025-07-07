from typing import Callable

import flax.linen as nn
from chex import PRNGKey
from jax import Array

from chromatix.elements.utils import register
from chromatix.typing import ScalarLike

from .. import functional as cf
from ..field import Field

__all__ = ["ThinLens", "FFLens", "DFLens"]


class ThinLens(nn.Module):
    """
    Applies a thin lens placed directly after the incoming ``Field``.
    This element returns the ``Field`` directly after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    The attributes ``f``, ``n``, and ``NA`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
    """

    f: ScalarLike | Callable[[PRNGKey], Array]
    n: ScalarLike | Callable[[PRNGKey], Array]
    NA: ScalarLike | Callable[[PRNGKey], Array] | None = None

    @nn.compact
    def __call__(self, field: Field) -> Field:
        f = register(self, "f")
        n = register(self, "n")
        NA = register(self, "NA")
        return cf.thin_lens(field, f, n, NA)


class FFLens(nn.Module):
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.
    This element returns the ``Field`` a distance ``f`` after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    The attributes ``f``, ``n``, and ``NA`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether to use IFFT (default is False, which uses FFT).
    """

    f: ScalarLike | Callable[[PRNGKey], Array]
    n: ScalarLike | Callable[[PRNGKey], Array]
    NA: ScalarLike | Callable[[PRNGKey], Array] | None = None
    inverse: bool = False

    @nn.compact
    def __call__(self, field: Field) -> Field:
        f = register(self, "f")
        n = register(self, "n")
        NA = register(self, "NA")
        return cf.ff_lens(field, f, n, NA, inverse=self.inverse)


class DFLens(nn.Module):
    """
    Applies a thin lens placed a distance ``d`` after the incoming ``Field``.
    This element returns the ``Field`` a distance ``f`` after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    The attributes ``d``, ``f``, ``n``, and ``NA`` can be learned by using
    ``chromatix.utils.trainable``.

    Attributes:
        d: Distance from the incoming ``Field`` to the lens.
        f: Focal length of the lens.
        n: Refractive index of the lens.
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether to use IFFT (default is False, which uses FFT).
    """

    d: ScalarLike | Callable[[PRNGKey], Array]
    f: ScalarLike | Callable[[PRNGKey], Array]
    n: ScalarLike | Callable[[PRNGKey], Array]
    NA: ScalarLike | Callable[[PRNGKey], Array] | None = None
    inverse: bool = False

    @nn.compact
    def __call__(self, field: Field) -> Field:
        d = register(self, "d")
        f = register(self, "f")
        n = register(self, "n")
        NA = register(self, "NA")
        return cf.df_lens(field, d, f, n, NA, inverse=self.inverse)
