import flax.linen as nn
from typing import Callable, Optional, Union
from chromatix import Field
from chex import PRNGKey
import chromatix.functional as cf

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

    For more details, see ``chromatix.functional.propagate``.

    Attributes:
        n: Refractive index.
        N_pad: If provided, the padding for propagation (will be used as both
            height and width padding). Otherwise, the padding will be
            automatically calculated in ``chromatix.functional.propagate``.
        method: The propagation method for ``chromatix.functional.propagate``.
        mode: Defines the cropping of the output in
            ``chromatix.functional.propagate``.
    """

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
