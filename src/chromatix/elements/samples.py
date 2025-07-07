from typing import Callable

import flax.linen as nn
from chex import PRNGKey
from jax import Array

from chromatix.typing import ArrayLike

from ..field import ScalarField
from ..functional.samples import thin_sample
from .utils import register

__all__ = ["ThinSample"]


class ThinSample(nn.Module):
    """
    Perturbs an incoming ``ScalarField`` as if it went through a thin sample
    object with a given ``absorption``, refractive index change ``dn`` and of
    a given ``thickness`` in the same units as the spectrum of the incoming
    ``ScalarField``.

    The sample is supposed to follow the thin sample approximation, so the
    sample perturbation is calculated as:
    ``exp(1j * 2 * pi * (dn + 1j * absorption) * thickness / lambda)``.

    Returns a ``ScalarField`` with the result of the perturbation.

    Attributes:
        field: The complex field to be perturbed.
        absorption: The sample absorption defined as ``(B... H W 1 1)`` array
        dn: Sample refractive index change ``(B... H W 1 1)`` array
        thickness: Thickness at each sample location as array broadcastable
            to ``(B... H W 1 1)``
    """

    absorption: ArrayLike | Callable[[PRNGKey], Array]
    dn: ArrayLike | Callable[[PRNGKey], Array]
    thickness: ArrayLike | Callable[[PRNGKey], Array]

    @nn.compact
    def __call__(self, field: ScalarField) -> ScalarField:
        absorption = register(self, "absorption")
        dn = register(self, "dn")
        thickness = register(self, "thickness")
        return thin_sample(field, absorption, dn, thickness)
