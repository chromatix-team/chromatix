from typing import Callable, Literal

import flax.linen as nn
from chex import PRNGKey
from jax import Array

from chromatix.elements.utils import Trainable, register
from chromatix.field import crop, pad
from chromatix.typing import ArrayLike

from ..field import Field
from ..functional import (
    asm_propagate,
    compute_asm_propagator,
    compute_transfer_propagator,
    kernel_propagate,
    transfer_propagate,
    transform_propagate,
)

__all__ = ["Propagate"]


class Propagate(nn.Module):
    """
    Free space propagation that can be placed after or between other elements.

    This element takes a ``Field`` as input and outputs a ``Field`` that has
    been propagated by a distance ``z``. Optionally, the index of refraction of
    the propagation medium can be learned.

    For example, if this element is constructed as:

    ```python
    from chromatix.elements import Propagate
    Propagate(n=1.33, method='transfer', mode='same')
    ```

    then this element has no trainable parameters, but if this element is
    constructed as:

    ```python
    from chromatix.elements import Propagate
    from chromatix.utils import trainable
    Propagate(n=trainable(1.33), method='transfer', mode='same')
    ```

    then this element has a trainable refractive index, initialized to 1.33.

    !!! warning
        The underlying propagation method now defaults to ``method=asm``,
        ``bandlimit=True`` and ``remove_evanescent=False``, which corresponds
        to bandlimited angular spectrum (BLAS) as proposed in "Band-Limited
        Angular Spectrum Method for Numerical Simulation of Free-Space
        Propagation in Far and Near Fields" (Matsumina et al., 2009).

    !!! warning
        By default this element caches the propagation kernel using the option
        ``cache_propagator``. Please be aware that this kernel gets placed
        inside the variables dict when initialising the model, so you'll have to
        split the dictionary into trainable parameters and non-trainable state.
        See the documentation Training Chromatix Models for more information on
        how to do this. If you would like to have a trainable propagation kernel
        with your own initialisation, see ``KernelPropagate`` which accepts a
        trainable ``propagator``.

    Attributes:
        z: Distance(s) to propagate.
        n: Refractive index.
        N_pad: The padding for propagation (will be used as both height and
            width padding). To automatically calculate the padding, use padding
            calculation functions from  ``chromatix.functional``. This must be
            passesd outside of a ``jax.jit``. Defaults to 0 (no padding), which
            will cause circular convolutions (edge artifacts) when propagating.
        cval: The value to pad with if ``N_pad`` is greater than 0. Defaults
            to 0.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        method: The propagation method, which can be "transform", "transfer",
            or "asm". Defaults to "asm", which is propagation without the
            Fresnel approximation.
        bandlimit: Whether to bandlimit the field before propagation for "asm".
            Defaults to True.
        remove_evanescent: Whether to remove evanescent waves when using the
            "asm" method. Defaults to False.
        mode: Defines the cropping of the output if the method is NOT
            "transform". Defaults to "same", which returns a ``Field`` of the
            same shape, unlike the functional methods.
        cache_propagator: Whether to compute and store the propagation kernel
            or not. If True, ``z`` and ``n`` cannot be trainable. Defaults
            to ``True``.
    """

    z: ArrayLike | Callable[[PRNGKey], Array]
    n: ArrayLike | Callable[[PRNGKey], Array]
    N_pad: int = 0
    cval: float = 0
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0)
    method: Literal["transform", "transfer", "asm"] = "asm"
    bandlimit: bool = True
    remove_evanescent: bool = False
    mode: Literal["full", "same"] = "same"
    cache_propagator: bool = True

    @nn.compact
    def __call__(self, field: Field) -> Field:
        if self.cache_propagator and (
            isinstance(self.z, Trainable) or isinstance(self.n, Trainable)
        ):
            raise ValueError("Cannot cache propagation kernel if z or n are trainable.")
        if self.cache_propagator and self.method not in ["transfer", "asm"]:
            raise ValueError("Can only cache kernel for 'transfer' or 'asm' methods.")
        z = register(self, "z")
        n = register(self, "n")
        if self.cache_propagator:
            field = pad(field, self.N_pad, cval=self.cval)
            propagator_args = (
                field,
                z,
                n,
                self.kykx,
            )
            if self.method == "transfer":
                propagator = self.variable(
                    "state",
                    "kernel",
                    lambda: compute_transfer_propagator(*propagator_args),
                )
            elif self.method == "asm":
                propagator = self.variable(
                    "state",
                    "kernel",
                    lambda: compute_asm_propagator(
                        *propagator_args,
                        remove_evanescent=self.remove_evanescent,
                        bandlimit=self.bandlimit,
                    ),
                )
            else:
                raise NotImplementedError
            field = kernel_propagate(field, propagator.value)
            if self.mode == "same":
                field = crop(field, self.N_pad)
            return field
        if self.method == "transform":
            return transform_propagate(
                field,
                z,
                n,
                N_pad=self.N_pad,
                cval=self.cval,
            )
        elif self.method == "transfer":
            return transfer_propagate(
                field,
                z,
                n,
                N_pad=self.N_pad,
                cval=self.cval,
                kykx=self.kykx,
                mode=self.mode,
            )
        elif self.method == "asm":
            return asm_propagate(
                field,
                z,
                n,
                N_pad=self.N_pad,
                cval=self.cval,
                kykx=self.kykx,
                mode=self.mode,
                remove_evanescent=self.remove_evanescent,
                bandlimit=self.bandlimit,
            )
        else:
            raise NotImplementedError(
                "Method must be one of: 'transform', 'transfer', 'asm'"
            )


class KernelPropagate(nn.Module):
    """
    Free space propagation with a precomputed propagation kernel.

    This element takes a ``Field`` as input and outputs a ``Field`` that has
    been propagated by a distance that is already defined by a propagation
    kernel. Optionally, this kernel can be a learned parameter using
    ``chromatix.utils.trainable``.

    All attributes other than the ``propagator`` and ``mode`` will be
    sent as arguments to the propagation kernel initialization function if
    ``propagator`` is trainable.

    Attributes:
        propagator: The propagation kernel to use. Can be trainable.
        z: Distance(s) to propagate. Defaults to None.
        n: Refractive index. Defaults to None.
        N_pad: The padding for propagation (will be used as both height and
            width padding). To automatically calculate the padding, use padding
            calculation functions from  ``chromatix.functional``. This must be
            passesd outside of a ``jax.jit``. Defaults to 0 (no padding), which
            will cause circular convolutions (edge artifacts) when propagating.
        cval: The value to pad with if ``N_pad`` is greater than 0. Defaults
            to 0.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format [ky, kx].
        mode: Defines the cropping of the output if the method is NOT
            "transform". Defaults to "same", which returns a ``Field`` of the
            same shape, unlike the functional methods.
    """

    propagator: ArrayLike | Callable[[PRNGKey], Array]
    z: ArrayLike | None = None
    n: ArrayLike | None = None
    N_pad: int = 0
    cval: float = 0
    kykx: ArrayLike | tuple[float, float] = (0.0, 0.0)
    mode: Literal["full", "same"] = "same"

    @nn.compact
    def __call__(self, field: Field) -> Field:
        field = pad(field, self.N_pad, cval=self.cval)
        propagator = register(
            self,
            "propagator",
            field,
            self.z,
            self.n,
            self.kykx,
        )

        field = kernel_propagate(field, propagator)
        if self.mode == "same":
            field = crop(field, self.N_pad)
        return field
