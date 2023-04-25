from flax import linen as nn
from typing import Optional


def register(
    module: nn.Module,
    name: str,
    *args,
    param_dict_name: Optional[str] = None,
    learnable: Optional[bool] = None,
):
    try:
        init = getattr(module, name)
    except AttributeError:
        print("Variable does not exist.")

    if learnable is None:
        try:
            learnable = name in getattr(module, "learnables")
        except AttributeError:
            print("Need either self.learnables or learnable keyword.")

    if param_dict_name is None:
        param_dict_name = f"_{name}"

    if learnable:
        return module.param(param_dict_name, parse_init(init), *args)
    else:
        return module.variable(
            "fixed_params",
            param_dict_name,
            parse_init(init),
            module.make_rng("params"),
            *args,
        ).value


def parse_init(x):
    def init(*args):
        return x

    return x if callable(x) else init

def trainable(x):
    
@dataclass
class Trainable:
    val:
    