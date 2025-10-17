import jax.numpy as jnp
from chex import assert_equal_shape
from jaxtyping import Array, Float, ScalarLike

from chromatix import (
    Absorbing,
    Sample,
    ScalarField,
    Scattering,
    Volume,
)
from chromatix.core.field import grid
from chromatix.functional.samples import multislice_thick_sample, thin_sample
from chromatix.utils import _broadcast_dx_to_grid

__all__ = [
    "ThinSample",
    "ClearThinSample",
    "MultisliceThickSample",
    "ClearMultisliceThickSample",
]


class ThinSample(Sample, Absorbing, Scattering, strict=True):
    """
    Perturbs an incoming ``ScalarField`` as if it went through a thin sample
    object with a given ``absorption``, refractive index change ``dn`` from
    the refractive index of the medium and of a given ``thickness`` in the same
    units as the spectrum of the incoming ``ScalarField``.

    The sample is supposed to follow the thin sample approximation, so the
    sample perturbation is calculated as:
    ``exp(1j * 2 * pi * (dn + 1j * absorption) * thickness / lambda)``.

    Returns a ``ScalarField`` with the result of the perturbation.

    Attributes:
        absorption: The sample's absorption defined as a 2D array of shape
            ``(height width)``.
        dn: The sample's isotropic refractive index change as a 2D array of
            shape ``(height width)``.
        dx: The spacing of each pixel of the sample as a scalar in units of
            distance or as a 1D array of shape `(2,)` in the format `(y x)`.
        thickness: Thickness in units of distance as a scalar or array
            broadcastable to ``(height width)`` (thickness at each sample
            location).
    """

    absorption: Float[Array, "h w"]
    dn: Float[Array, "h w"]
    dx: ScalarLike | Float[Array, "2"]
    thickness: ScalarLike | Float[Array, "h w"]

    def __init__(
        self,
        absorption: Float[Array, "h w"],
        dn: Float[Array, "h w"],
        dx: ScalarLike | Float[Array, "2"],
        thickness: ScalarLike | Float[Array, "h w"],
    ):
        assert_equal_shape(
            [absorption, dn],
            custom_message="Absorption and refractive index change arrays must have same shape",
        )
        self.absorption = absorption
        self.dn = dn
        self.dx = _broadcast_dx_to_grid(dx, 1).squeeze()
        self.thickness = thickness

    def __call__(self, field: ScalarField) -> ScalarField:
        self._verify_matching_spacing(field)
        return thin_sample(field, self.absorption, self.dn, self.thickness)

    @property
    def shape(self) -> tuple[int, int]:
        return self.absorption.shape

    @property
    def ndim(self) -> int:
        return self.absorption.ndim

    @property
    def grid(self) -> Array:
        return grid(self.shape, self.dx)


class ClearThinSample(Sample, Scattering, strict=True):
    """
    Perturbs an incoming ``ScalarField`` as if it went through a clear
    thin sample object with a given refractive index change ``dn`` from the
    refractive index of the medium and of a given ``thickness`` in the same
    units as the spectrum of the incoming ``ScalarField``.

    The sample is supposed to follow the thin sample approximation, so the
    sample perturbation is calculated as:
    ``exp(1j * 2 * pi * dn * thickness / lambda)``.

    Returns a ``ScalarField`` with the result of the perturbation.

    Attributes:
        dn: The sample's isotropic refractive index change as a 2D array of
            shape ``(height width)``.
        dx: The spacing of each pixel of the sample as a scalar in units of
            distance or as a 1D array of shape `(2,)` in the format `(y x)`.
        thickness: Thickness in units of distance as a scalar or array
            broadcastable to ``(height width)`` (thickness at each sample
            location).
    """

    dn: Float[Array, "h w"]
    dx: ScalarLike | Float[Array, "2"]
    thickness: ScalarLike | Float[Array, "h w"]

    def __init__(
        self,
        dn: Float[Array, "h w"],
        dx: ScalarLike | Float[Array, "2"],
        thickness: ScalarLike | Float[Array, "h w"],
    ):
        self.dn = dn
        self.dx = _broadcast_dx_to_grid(dx, 1).squeeze()
        self.thickness = thickness

    def __call__(self, field: ScalarField) -> ScalarField:
        self._verify_matching_spacing(field)
        return thin_sample(field, jnp.zeros_like(self.dn), self.dn, self.thickness)

    @property
    def shape(self) -> tuple[int, int]:
        return self.dn.shape

    @property
    def ndim(self) -> int:
        return self.dn.ndim

    @property
    def grid(self) -> Array:
        return grid(self.shape, self.dx)


class MultisliceThickSample(Sample, Absorbing, Scattering, Volume, strict=True):
    """
    Perturbs incoming ``ScalarField`` as if it went through a thick sample. The
    thick sample is modeled as being made of many thin slices each of a given
    thickness. The ``absorption_stack`` and ``dn_stack`` contain the absorbance
    and change in isotropic refractive index from the refractive index of the
    medium of each sample slice. Expects that the same sample is being applied
    to all elements across the batch of the incoming ``Field``.

    By default, a ``propagator`` is calculated inside the function.
    After passing through all slices, the field is propagated backwards
    to the center of the stack, or by the distances specified by
    ``reverse_propagate_distance`` if provided.

    Returns a ``Field`` with the result of the perturbation.

    !!! warning
        The underlying propagation method now defaults to the angular spectrum
        method (ASM) with ``bandlimit=False`` and ``remove_evanescent=False``.

    Attributes:
        absorption: The sample's absorption per voxel for each slice defined as
            a 3D array of shape ``(depth height width)``, where ``depth`` is the
            total number of slices.
        dn: The sample's isotropic refractive index change for each slice as a
            3D array of shape ``(depth height width)``. Shape must be the same
            that for ``absorption_stack``.
        n: Average refractive index of the sample.
        dx: The spacing of each pixel of the sample as a scalar in units of
            distance or as a 1D array of shape `(2,)` in the format `(y x)`.
        thickness: How far to propagate for each slice in units of distance.
        NA: If provided, will be used to define the numerical aperture (limiting
            the captured frequencies) of the lens that is imaging the center of
            the volume. If not provided (the default case), this function will
            return the scattered field directly which may have undesirable high
            frequencies.
    """

    absorption: Float[Array, "d h w"]
    dn: Float[Array, "d h w"]
    n: ScalarLike
    dx: ScalarLike | Float[Array, "2"]
    thickness: ScalarLike
    NA: ScalarLike | None

    def __init__(
        self,
        absorption: Float[Array, "d h w"],
        dn: Float[Array, "d h w"],
        n: ScalarLike,
        dx: ScalarLike | Float[Array, "2"],
        thickness: ScalarLike,
        NA: ScalarLike | None = None,
    ):
        self.absorption = absorption
        self.dn = dn
        self.n = n
        self.dx = _broadcast_dx_to_grid(dx, 1).squeeze()
        self.thickness = thickness
        self._verify_scalar_thickness()
        self.NA = NA

    def __call__(
        self,
        field: ScalarField,
        kykx: Float[Array, "2"] = (0.0, 0.0),
        pad_width: int = 0,
        reverse_propagate_distance: ScalarLike | None = None,
        return_stack: bool = False,
        remove_evanescent: bool = False,
        bandlimit: bool = False,
    ) -> ScalarField:
        """
        Args:
            field: The complex field to be perturbed.
            kykx: If provided, defines the orientation of the propagation.
                Should be an array of shape `(2,)` in the format ``ky kx``.
            pad_width: An integer defining the pad length for the
                propagation FFT (NOTE: should not be a `jax` ``Array``,
                otherwise a ConcretizationError will arise when
                traced!). You can use padding calculator utilities from
                ``chromatix.functional.propagation`` to estimate the padding.
            reverse_propagate_distance: If provided, propagates field at the end
                backwards by this amount from the top of the stack. By default,
                field is propagated backwards to the middle of the sample.
            return_stack: If ``True``, returns the 3D stack of intermediate
                scattered fields at each plane of the thick sample. This 3D
                stack is returned as a ``Field`` where the innermost batch
                dimension is the number of planes in the provided ``dn_stack``/
                ``absorption_stack`` instead of the field defocused to the middle of
                the sample after scattering through the whole sample. If ``True``,
                ``reverse_propagate_distance`` is ignored. Defaults to ``False``.
            remove_evanescent: If ``True``, removes evanescent waves. Defaults to
                ``False``.
            bandlimit: Whether to bandlimit the field before propagation.
                Defaults to ``False``.
        """
        self._verify_matching_spacing(field)
        return multislice_thick_sample(
            field,
            self.absorption,
            self.dn,
            self.n,
            self.thickness,
            pad_width,
            NA=self.NA,
            kykx=kykx,
            reverse_propagate_distance=reverse_propagate_distance,
            return_stack=return_stack,
            remove_evanescent=remove_evanescent,
            bandlimit=bandlimit,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.absorption.shape

    @property
    def ndim(self) -> int:
        return self.absorption.ndim

    @property
    def grid(self) -> Array:
        return grid(self.shape, self.dx)

    @property
    def num_planes(self) -> int:
        return self.absorption.shape[0]


class ClearMultisliceThickSample(Sample, Scattering, Volume, strict=True):
    """
    Perturbs incoming ``ScalarField`` as if it went through a clear but
    scattering thick sample. The thick sample is modeled as being made of many
    thin slices each of a given thickness. The ``dn_stack`` contains the change
    in isotropic refractive index from the refractive index of the medium of
    each sample slice. Expects that the same sample is being applied to all
    elements across the batch of the incoming ``Field``.

    By default, a ``propagator`` is calculated inside the function.
    After passing through all slices, the field is propagated backwards
    to the center of the stack, or by the distances specified by
    ``reverse_propagate_distance`` if provided.

    Returns a ``Field`` with the result of the perturbation.

    !!! warning
        The underlying propagation method now defaults to the angular spectrum
        method (ASM) with ``bandlimit=False`` and ``remove_evanescent=False``.

    Attributes:
        dn: The sample's isotropic refractive index change per voxel for
            each slice as a 3D array of shape ``(depth height width)`` where
            ``depth`` is the total number of slices.
        n: Average refractive index of the sample.
        dx: The spacing of each pixel of the sample as a scalar in units of
            distance or as a 1D array of shape `(2,)` in the format `(y x)`.
        thickness: How far to propagate for each slice in units of distance.
        NA: If provided, will be used to define the numerical aperture (limiting
            the captured frequencies) of the lens that is imaging the center of
            the volume. If not provided (the default case), this function will
            return the scattered field directly which may have undesirable high
            frequencies.
    """

    dn: Float[Array, "d h w"]
    n: ScalarLike
    dx: ScalarLike | Float[Array, "2"]
    thickness: ScalarLike
    NA: ScalarLike | None

    def __init__(
        self,
        dn: Float[Array, "d h w"],
        n: ScalarLike,
        dx: ScalarLike | Float[Array, "2"],
        thickness: ScalarLike,
        NA: ScalarLike | None = None,
    ):
        self.dn = dn
        self.n = n
        self.dx = _broadcast_dx_to_grid(dx, 1).squeeze()
        self.thickness = thickness
        self._verify_scalar_thickness()
        self.NA = NA

    def __call__(
        self,
        field: ScalarField,
        kykx: Float[Array, "2"] = (0.0, 0.0),
        pad_width: int = 0,
        reverse_propagate_distance: ScalarLike | None = None,
        return_stack: bool = False,
        remove_evanescent: bool = False,
        bandlimit: bool = False,
    ) -> ScalarField:
        """
        Args:
            field: The complex field to be perturbed.
            kykx: If provided, defines the orientation of the propagation.
                Should be an array of shape `(2,)` in the format ``ky kx``.
            pad_width: An integer defining the pad length for the
                propagation FFT (NOTE: should not be a `jax` ``Array``,
                otherwise a ConcretizationError will arise when
                traced!). You can use padding calculator utilities from
                ``chromatix.functional.propagation`` to estimate the padding.
            reverse_propagate_distance: If provided, propagates field at the end
                backwards by this amount from the top of the stack. By default,
                field is propagated backwards to the middle of the sample.
            return_stack: If ``True``, returns the 3D stack of intermediate
                scattered fields at each plane of the thick sample. This 3D
                stack is returned as a ``Field`` where the innermost batch
                dimension is the number of planes in the provided ``dn_stack``/
                ``absorption_stack`` instead of the field defocused to the middle of
                the sample after scattering through the whole sample. If ``True``,
                ``reverse_propagate_distance`` is ignored. Defaults to ``False``.
            remove_evanescent: If ``True``, removes evanescent waves. Defaults to
                ``False``.
            bandlimit: Whether to bandlimit the field before propagation.
                Defaults to ``False``.
        """
        self._verify_matching_spacing(field)
        return multislice_thick_sample(
            field,
            jnp.zeros_like(self.dn),
            self.dn,
            self.n,
            self.thickness,
            pad_width,
            NA=self.NA,
            kykx=kykx,
            reverse_propagate_distance=reverse_propagate_distance,
            return_stack=return_stack,
            remove_evanescent=remove_evanescent,
            bandlimit=bandlimit,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dn.shape

    @property
    def ndim(self) -> int:
        return self.dn.ndim

    @property
    def grid(self) -> Array:
        return grid(self.shape, self.dx)

    @property
    def num_planes(self) -> int:
        return self.dn.shape[0]
