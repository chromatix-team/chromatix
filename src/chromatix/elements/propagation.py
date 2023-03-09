import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Literal, Optional, Union
from chromatix import Field
from chex import PRNGKey, Array
from chromatix.functional import (
    transform_propagate,
    transfer_propagate,
    exact_propagate,
    kernel_propagate,
    compute_transfer_propagator,
    compute_exact_propagator,
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

    This element can cache the propagation kernel using the option
    ``cache_propagator``. If you would like to have a trainable propagation
    kernel, see ``KernelPropagate`` which accepts a trainable ``propagator``.

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
            or "exact." Defaults to "exact", which is propagation without the
            Fresnel approximation.
        mode: Defines the cropping of the output if the method is "transfer" or
            "exact". Defaults to "same", which returns a ``Field`` of the same
            shape, unlike the functional methods.
        cache_propagator: Whether to compute and store the propagation kernel
            or not. If True, ``z`` and ``n`` cannot be trainable. Defaults
            to True.
    """

    z: Union[float, Array, Callable[[PRNGKey], Array]]
    n: Union[float, Callable[[PRNGKey], Array]]
    N_pad: int = 0
    cval: float = 0
    kykx: Array = jnp.zeros((2,))
    method: Literal["transform", "transfer", "exact"] = "exact"
    mode: Literal["full", "same"] = "same"
    cache_propagator: bool = True
    loop_axis: Optional[int] = None

    @nn.compact
    def __call__(self, field: Field) -> Field:
        if self.cache_propagator and (
            isinstance(self.z, Callable) or isinstance(self.n, Callable)
        ):
            raise ValueError("Cannot cache propagation kernel if z or n are trainable.")
        if self.cache_propagator and self.method not in ["transfer", "exact"]:
            raise ValueError("Can only cache kernel for 'transfer' or 'exact' methods.")
        z = self.param("_z", self.z) if isinstance(self.z, Callable) else self.z
        n = self.param("_n", self.n) if isinstance(self.n, Callable) else self.n
        if self.cache_propagator:
            propagator_args = (
                field.shape,
                field.dx,
                field.spectrum,
                z,
                n,
                self.N_pad,
                self.kykx,
            )
            if self.method == "transfer":
                propagator = self.variable(
                    "propagation",
                    "kernel",
                    lambda: compute_transfer_propagator(*propagator_args),
                )
            elif self.method == "exact":
                propagator = self.variable(
                    "propagation",
                    "kernel",
                    lambda: compute_exact_propagator(*propagator_args),
                )
            return kernel_propagate(
                field,
                propagator.value,
                self.N_pad,
                self.cval,
                self.loop_axis,
                self.mode,
            )
        if self.method == "transform":
            return transform_propagate(
                field,
                z,
                n,
                N_pad=self.N_pad,
                loop_axis=self.loop_axis,
            )
        elif self.method == "transfer":
            return transfer_propagate(
                field,
                z,
                n,
                N_pad=self.N_pad,
                cval=self.cval,
                kykx=self.kykx,
                loop_axis=self.loop_axis,
                mode=self.mode,
            )
        elif self.method == "exact":
            return exact_propagate(
                field,
                z,
                n,
                N_pad=self.N_pad,
                cval=self.cval,
                kykx=self.kykx,
                loop_axis=self.loop_axis,
                mode=self.mode,
            )
        else:
            raise NotImplementedError(
                "Method must be one of 'transform', 'transfer', or 'exact'."
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
        mode: Defines the cropping of the output if the method is "transfer" or
            "exact". Defaults to "same", which returns a ``Field`` of the same
            shape, unlike the functional methods.
    """

    propagator: Union[Array, Callable[[PRNGKey], Array]]
    z: Optional[Union[float, Array]] = None
    n: Optional[float] = None
    N_pad: int = 0
    cval: float = 0
    kykx: Array = jnp.zeros((2,))
    mode: Literal["full", "same"] = "same"
    loop_axis: Optional[int] = None

    @nn.compact
    def __call__(self, field: Field) -> Field:
        if isinstance(self.propagator, Callable):
            propagator = self.param(
                "_propagator",
                self.propagator,
                field.shape,
                field.dx,
                field.spectrum,
                self.z,
                self.n,
                self.N_pad,
                self.kykx,
            )
        else:
            propagator = self.propagator
        return kernel_propagate(
            field,
            propagator,
            self.N_pad,
            self.cval,
            self.loop_axis,
            self.mode,
        )
