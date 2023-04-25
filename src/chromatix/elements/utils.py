from flax import linen as nn
from typing import Optional, Any
from dataclasses import dataclass


def register(
    module: nn.Module,
    name: str,
    *args,
):
    try:
        init = getattr(module, name)
    except AttributeError:
        print("Variable does not exist.")

    if isinstance(init, Trainable):
        return module.param(f"_{name}", parse_init(init.val), *args)
    else:
        return module.variable(
            "fixed_params",
            f"_{name}",
            parse_init(init),
            module.make_rng("params"),
            *args,
        ).value


def parse_init(x):
    def init(*args):
        return x

    return x if callable(x) else init


def trainable(x):
    return Trainable(x)


@dataclass
class Trainable:
    val: Any
