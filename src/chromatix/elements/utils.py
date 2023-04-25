from flax import linen as nn
from chromatix.utils import Trainable


def register(
    module: nn.Module,
    name: str,
    *args,
):
    """Registers the parameter `self.{name}` as a Flax parameter or variable depending
    on whether the parameter is of type `Trainable`. Only used for internal ease-of-use.

    Name in Flax's parameterdict becomes `_{name}`, and if variable under collection
    `state`. Supports initializing both with callables (*args are passed as
    arguments) and fixed values.

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
            None,
            *args,
        ).value


def parse_init(x):
    def init(*args):
        return x

    return x if callable(x) else init
