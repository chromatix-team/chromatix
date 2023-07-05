from flax import linen as nn
from dataclasses import dataclass
from typing import Any, Callable
from chex import PRNGKey, Array


@dataclass
class Trainable:
    """
    Wrapper class to signal to a Chromatix element that ``val`` should be the
    initialization for a trainable parameter.
    """

    val: Any


def trainable(x: Any, rng: bool = True) -> Trainable:
    """
    Returns ``x`` wrapped in a ``Trainable`` object to signal to a
    Chromatix element that ``x`` should be used to initialize a trainable
    parameter. If ``x`` is already a function, then this function will be
    used as the initializer. If ``x`` is a function that does not accept a
    ``jax.random.PRNGKey``, then setting ``rng`` to ``False`` will wrap ``x``
    so that the arguments for ``x`` are accepted after the ``PRNGKey`` argument.
    This is useful since many Chromatix functions you might want to use as
    initialization functions don't accept ``PRNGKey`` arguments. Note that this
    argument does not matter if ``x`` is already an ``Array`` that can be used
    as an initialization directly.

    When a supported Chromatix element is constructed with this wrapper as its
    attribute, it will automatically turn that attribute into a parameter to
    be optimized. Thus, this function is a convenient way to set the attribute
    of an optical element in Chromatix as a trainable parameter initialized
    to the value defined by ``x``. Any element that has potentially trainable
    parameters will be documented as such.

    For example, we can initialize a trainable phase mask (allowing for the
    optimization of the pixels of the phase mask for arbitrary tasks) with this
    function in two different ways:

    ```python
    from chromatix.utils import trainable
    from chromatix.functional import potato_chip
    from chromatix.elements import PhaseMask

    phase_mask = PhaseMask(
        phase=trainable(
            potato_chip(
                shape=(3840, 3840),
                spacing=0.3,
                wavelength=0.5,
                n=1.33,
                f=100,
                NA=0.8
            )
        )
    )
    params = phase_mask.init()
    ```

    This example directly calls ``potato_chip`` to create a trainable phase
    mask with the given shape. If there is a mismatch between the shape of an
    incoming ``Field`` and the shape of the ``phase``, then an error will occur
    at runtime. For many applications, the shape of the ``Field`` will be known
    and fixed, so this style of initialization is convenient. The second way is
    slightly more complex but also more robust to these shape issues, and does
    not require declaring the shapes twice:

    ```python
    from chromatix.utils import trainable
    from chromatix.functional import potato_chip
    from chromatix.elements import PhaseMask
    from functools import partial

    phase_mask = PhaseMask(
        phase=trainable(
            partial(
                potato_chip, spacing=0.3, wavelength=0.5, n=1.33, f=100, NA=0.8
            ),
            rng=False
        )
    )
    ```

    When ``PhaseMask`` initializes its parameters, it automatically passes
    a ``jax.random.PRNGKey`` and the spatial shape of the input ``Field``,
    which were ignored in the previous example because the initial ``phase``
    was an ``Array`` constructed by ``potato_chip``. This example uses
    ``functools.partial`` to create a phase mask initialization function that
    only accepts a shape, which is wrapped by ``trainable`` to also accept
    a ``jax.random.PRNGKey`` as its first argument. Now, when ``PhaseMask``
    initializes its parameters, it will call this initialization function,
    which uses the shape of the input ``Field`` to calculate the initial phase.
    This matches the signature of the common ``jax.nn.initializers``, which
    also accept a ``jax.random.PRNGKey`` and a shape.

    Args:
        x: The value that will be used to initialize the trainable
            parameter.
        rng: Whether the initializer function ``x`` needs a ``PRNGKey`` or not.
            If ``True``, assumes that the function ``x`` has a ``PRNGKey`` as
            its first argument, and does not modify ``x``. If ``False``, wraps
            the initializer function ``x`` to ignore the ``PRNGKey`` argument
            passed by Flax. If ``x`` is not callable, then this argument doesn't
            matter and is ignored. Defaults to ``True``.

    Returns:
        A function that takes a ``jax.random.PRNGKey`` as its first parameter.
    """
    init = x
    if callable(x) and not rng:

        def no_rng_x(key: PRNGKey, *args, **kwargs) -> Array:
            return x(*args, **kwargs)

        init = no_rng_x
    return Trainable(init)


def register(
    module: nn.Module,
    name: str,
    *args,
) -> Any:
    """
    Registers the attribute `module.{name}` as a Flax parameter or variable
    depending on whether the attribute is of type `Trainable`. Only for internal
    use in Chromatix elements.

    The name of the parameter becomes `_{name}` in the Flax variable dictionary
    (either under `"params"` or `"state"`). Supports initializing both with
    callables (*args are passed as arguments) and fixed values.
    """
    try:
        init = getattr(module, name)
    except AttributeError:
        print("Variable does not exist.")

    if isinstance(init, Trainable):
        return module.param(f"_{name}", parse_init(init.val), *args)
    else:
        return module.variable(
            "state",
            f"_{name}",
            parse_init(init),
            *args,
        ).value


def parse_init(x: Any) -> Callable:
    """
    Returns a function that returns ``x`` if ``x`` is not a function, otherwise
    simply returns the function ``x``.

    Args:
        x: The value or function that initializes a variable.

    Returns:
        A function that will be used as an initializer to a variable.
    """

    def init(*args) -> Any:
        return x

    return x if callable(x) else init
