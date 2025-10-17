from typing import Literal

import equinox as eqx
from jaxtyping import Array, Float, ScalarLike

from chromatix import Field, crop, pad
from chromatix.functional import (
    asm_propagate,
    compute_asm_propagator,
    compute_transfer_propagator,
    kernel_propagate,
    transfer_propagate,
    transform_propagate,
)
from chromatix.typing import z

__all__ = ["Propagate", "KernelPropagate"]


class Propagate(eqx.Module):
    """
    Free space propagation that can be placed after or between other elements.

    This element takes a ``Field`` as input and outputs a ``Field`` that has
    been propagated by a distance ``z``. Initialization of this element requires a
    ``Field`` to construct the propagation kernels.

    For example, this element can be constructed as:

    ```python
    from chromatix.elements import Propagate
    from chromatix.functional import plane_wave
    field = plane_wave(...)
    propagation = Propagate(field, z=1e3, n=1.33, method='asm')
    ```

    !!! warning
        The underlying propagation method now defaults to ``method=asm``,
        ``bandlimit=True`` and ``remove_evanescent=False``, which corresponds
        to bandlimited angular spectrum (BLAS) as proposed in "Band-Limited
        Angular Spectrum Method for Numerical Simulation of Free-Space
        Propagation in Far and Near Fields" (Matsumina et al., 2009).

    !!! warning
        By default this element caches the propagation kernel using the option
        ``cache_propagator``. If you would like to have a propagation kernel
        with your own initialization, see ``KernelPropagate`` which accepts a
        custom ``propagator``.

    Attributes:
        z: How far to propagate as a scalar value or a 1D array in units of distance.
        n: Refractive index.
        pad_width: The padding for propagation (will be used as both height and
            width padding). To automatically calculate the padding, use padding
            calculation functions from  ``chromatix.functional``. This must be
            passesd outside of a ``jax.jit``. Defaults to 0 (no padding), which
            will cause circular convolutions (edge artifacts) when propagating.
        cval: The value to pad with if ``pad_width`` is greater than 0. Defaults
            to 0.
        kykx: If provided, defines the orientation of the propagation. Should
            be an array of shape `[2,]` in the format `[ky, kx]`.
        method: The propagation method, which can be "transform", "transfer",
            or "asm". Defaults to "asm", which is propagation as a Fourier
            convolution (two Fourier transforms) without the Fresnel
            approximation.
        bandlimit: Whether to bandlimit the field before propagation for "asm".
            Defaults to True.
        remove_evanescent: Whether to remove evanescent waves when using the
            "asm" method. Defaults to False.
        mode: Defines the cropping of the output if the method is NOT
            "transform". Defaults to "same", which returns a ``Field`` of the
            same shape, unlike the functional methods.
        cache_propagator: Whether to precompute and store the propagation kernel
            or not. If ``True``, none of the other attributes will be used after
            initializing the propagation kernel and the propagation kernel will
            only be computed once at initialization. Defaults to ``True``.
    """

    z: ScalarLike | Float[Array, "z"]
    n: ScalarLike
    pad_width: int = eqx.field(static=True)
    cval: ScalarLike
    kykx: Float[Array, "2"] | tuple[float, float]
    method: Literal["transform", "transfer", "asm"] = eqx.field(static=True)
    bandlimit: bool = eqx.field(static=True)
    remove_evanescent: bool = eqx.field(static=True)
    mode: Literal["full", "same"] = eqx.field(static=True)
    cache_propagator: bool = eqx.field(static=True)
    propagator: Array | None

    def __init__(
        self,
        field: Field,
        z: ScalarLike | Float[Array, "z"],
        n: ScalarLike,
        pad_width: int = 0,
        cval: float = 0,
        kykx: Float[Array, "2"] | tuple[float, float] = (0.0, 0.0),
        method: Literal["transform", "transfer", "asm"] = "asm",
        bandlimit: bool = True,
        remove_evanescent: bool = False,
        mode: Literal["full", "same"] = "same",
        cache_propagator: bool = True,
    ):
        self.z = z
        self.n = n
        self.pad_width = pad_width
        self.cval = cval
        self.kykx = kykx
        self.method = method
        self.bandlimit = bandlimit
        self.remove_evanescent = remove_evanescent
        self.mode = mode
        self.cache_propagator = cache_propagator
        if self.cache_propagator:
            field = pad(field, self.pad_width, cval=self.cval)
            propagator_args = (
                field,
                self.z,
                self.n,
                self.kykx,
            )
            if self.method == "transfer":
                self.propagator = compute_transfer_propagator(*propagator_args)
            elif self.method == "asm":
                self.propagator = compute_asm_propagator(
                    *propagator_args,
                    remove_evanescent=self.remove_evanescent,
                    bandlimit=self.bandlimit,
                )
            else:
                raise ValueError(
                    "Can only cache kernel for 'transfer' or 'asm' methods"
                )
        else:
            self.propagator = None

    def __call__(self, field: Field) -> Field:
        if self.cache_propagator:
            field = pad(field, self.pad_width, cval=self.cval)
            field = kernel_propagate(field, self.propagator)
            if self.mode == "same":
                field = crop(field, self.pad_width)
            return field
        if self.method == "transform":
            return transform_propagate(
                field,
                self.z,
                self.n,
                pad_width=self.pad_width,
                cval=self.cval,
            )
        elif self.method == "transfer":
            return transfer_propagate(
                field,
                self.z,
                self.n,
                pad_width=self.pad_width,
                cval=self.cval,
                kykx=self.kykx,
                mode=self.mode,
            )
        elif self.method == "asm":
            return asm_propagate(
                field,
                self.z,
                self.n,
                pad_width=self.pad_width,
                cval=self.cval,
                kykx=self.kykx,
                mode=self.mode,
                remove_evanescent=self.remove_evanescent,
                bandlimit=self.bandlimit,
            )
        else:
            raise ValueError("Method must be one of: 'transform', 'transfer', 'asm'")


class KernelPropagate(eqx.Module):
    """
    Free space propagation with a precomputed propagation kernel.

    This element takes a ``Field`` as input and outputs a ``Field`` that has
    been propagated by a distance that is already defined by a propagation
    kernel. Propagation is performed via Fourier convolution (two Fourier
    transforms).

    Attributes:
        propagator: The propagation kernel to use (can be created using e.g.
            [`compute_asm_propagator`](chromatix.functional.propagation.compute_asm_propagator)).
        z: Distance(s) to propagate. Defaults to None.
        n: Refractive index. Defaults to None.
        pad_width: The padding for propagation (will be used as both height and
            width padding). To automatically calculate the padding, use padding
            calculation functions from  ``chromatix.functional``. This must be
            passed outside of a ``jax.jit``. Defaults to 0 (no padding), which
            will cause circular convolutions (edge artifacts) when propagating.
        cval: The value to pad with if ``pad_width`` is greater than 0. Defaults
            to 0.
        kykx: If provided, defines the orientation of the propagation. Should be
            an array of shape `[2,]` in the format `[ky, kx]`.
        mode: Defines the cropping of the output if the method is NOT
            "transform". Defaults to "same", which returns a ``Field`` of the
            same shape, unlike the functional methods.
    """

    propagator: Array
    z: ScalarLike | Float[Array, "z"]
    n: ScalarLike
    pad_width: int = eqx.field(static=True)
    cval: ScalarLike
    kykx: Float[Array, "2"] | tuple[float, float]
    mode: Literal["full", "same"] = eqx.field(static=True)

    def __init__(
        self,
        propagator: Array,
        z: ScalarLike | Float[Array, "z"],
        n: ScalarLike,
        pad_width: int = 0,
        cval: float = 0,
        kykx: Float[Array, "2"] | tuple[float, float] = (0.0, 0.0),
        mode: Literal["full", "same"] = "same",
    ):
        self.propagator = propagator
        self.z = z
        self.n = n
        self.pad_width = pad_width
        self.cval = cval
        self.kykx = kykx
        self.mode = mode

    def __call__(self, field: Field) -> Field:
        field = pad(field, self.pad_width, cval=self.cval)
        field = kernel_propagate(field, self.propagator)
        if self.mode == "same":
            field = crop(field, self.pad_width)
        return field
