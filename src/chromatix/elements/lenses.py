import equinox as eqx
from jaxtyping import ScalarLike

from chromatix import Field

from .. import functional as cf

__all__ = ["ThinLens", "FFLens", "DFLens"]


class ThinLens(eqx.Module):
    """
    Applies a thin lens placed directly after the incoming ``Field``.
    This element returns the ``Field`` directly after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    Attributes:
        f: Focal length of the lens in units of distance.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
    """

    f: ScalarLike
    n: ScalarLike
    NA: ScalarLike | None

    def __init__(self, f: ScalarLike, n: ScalarLike, NA: ScalarLike | None = None):
        """
        Applies a thin lens placed directly after the incoming ``Field``.
        This element returns the ``Field`` directly after the lens.

        This element can be placed after any element that returns a ``Field`` or
        before any element that accepts a ``Field``.

        Args:
            f: Focal length of the lens in units of distance.
            n: The refractive index of the surrounding medium (assumed to be the
                same incoming and exiting).
            NA: If provided, the NA of the lens. By default, no pupil is applied
                to the incoming ``Field``.
        """
        self.f = f
        self.n = n
        self.NA = NA

    def __call__(self, field: Field) -> Field:
        return cf.thin_lens(field, self.f, self.n, self.NA)


class FFLens(eqx.Module):
    """
    Applies a thin lens placed a distance ``f`` after the incoming ``Field``.
    This element returns the ``Field`` a distance ``f`` after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    Attributes:
        f: Focal length of the lens.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether the field is passing forwards or backwards through
            the lens. If ``True``, the phase of the lens is conjugated.
            Defaults to ``False``.
    """

    f: ScalarLike
    n: ScalarLike
    NA: ScalarLike | None
    inverse: bool = eqx.field(static=True)

    def __init__(
        self,
        f: ScalarLike,
        n: ScalarLike,
        NA: ScalarLike | None = None,
        inverse: bool = False,
    ):
        """
        Applies a thin lens placed directly after the incoming ``Field``.
        This element returns the ``Field`` directly after the lens.

        This element can be placed after any element that returns a ``Field`` or
        before any element that accepts a ``Field``.

        Args:
            f: Focal length of the lens in units of distance.
            n: The refractive index of the surrounding medium (assumed to be the
                same incoming and exiting).
            NA: If provided, the NA of the lens. By default, no pupil is applied
                to the incoming ``Field``.
            inverse: Whether the field is passing forwards or backwards through
                the lens. If ``True``, the phase of the lens is conjugated.
                Defaults to ``False``.
        """
        self.f = f
        self.n = n
        self.NA = NA
        self.inverse = inverse

    def __call__(self, field: Field) -> Field:
        return cf.ff_lens(field, self.f, self.n, self.NA, inverse=self.inverse)


class DFLens(eqx.Module):
    """
    Applies a thin lens placed a distance ``d`` after the incoming ``Field``.
    This element returns the ``Field`` a distance ``f`` after the lens.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    Attributes:
        d: How far away the lens is from the incoming ``Field`` in units of distance.
        f: Focal length of the lens in units of distance.
        n: The refractive index of the surrounding medium (assumed to be the
            same incoming and exiting).
        NA: If provided, the NA of the lens. By default, no pupil is applied
            to the incoming ``Field``.
        inverse: Whether the field is passing forwards or backwards through
            the lens. If ``True``, the phase of the lens is conjugated.
            Defaults to ``False``.
    """

    d: ScalarLike
    f: ScalarLike
    n: ScalarLike
    NA: ScalarLike | None
    inverse: bool = eqx.field(static=True)

    def __init__(
        self,
        d: ScalarLike,
        f: ScalarLike,
        n: ScalarLike,
        NA: ScalarLike | None = None,
        inverse: bool = False,
    ):
        """
        Applies a thin lens placed a distance ``d`` after the incoming ``Field``.
        This element returns the ``Field`` a distance ``f`` after the lens.

        This element can be placed after any element that returns a ``Field`` or
        before any element that accepts a ``Field``.

        Args:
            d: How far away the lens is from the incoming ``Field`` in units
                of distance.
            f: Focal length of the lens in units of distance.
            n: The refractive index of the surrounding medium (assumed to be the
                same incoming and exiting).
            NA: If provided, the NA of the lens. By default, no pupil is applied
                to the incoming ``Field``.
            inverse: Whether the field is passing forwards or backwards through
                the lens. If ``True``, the phase of the lens is conjugated.
                Defaults to ``False``.
        """
        self.d = d
        self.f = f
        self.n = n
        self.NA = NA
        self.inverse = inverse

    def __call__(self, field: Field) -> Field:
        return cf.df_lens(field, self.d, self.f, self.n, self.NA, inverse=self.inverse)
