import flax.linen as nn
from typing import Callable, Optional, Union
from chromatix import Field
from chex import PRNGKey
import chromatix.functional as cf

__all__ = ["Propagate"]


class Propagate(nn.Module):
    n: Union[float, Callable[[PRNGKey], float]]
    N_pad: Optional[int] = None
    method: str = "transfer"
    mode: str = "same"
    loop_axis: Optional[int] = None

    def setup(self):
        self._n = self.param("_n", self.n) if isinstance(self.n, Callable) else self.n

    def __call__(self, field: Field, z: float) -> Field:
        return cf.propagate(
            field,
            z,
            self._n,
            method=self.method,
            mode=self.mode,
            N_pad=self.N_pad,
            loop_axis=self.loop_axis,
        )
