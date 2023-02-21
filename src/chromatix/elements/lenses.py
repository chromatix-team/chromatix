import flax.linen as nn
from typing import Callable, Optional, Union
from chex import PRNGKey
from chromatix import Field
import chromatix.functional as cf

__all__ = ["ThinLens", "FFLens", "DFLens"]


class ThinLens(nn.Module):
    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Optional[Union[float, Callable[[PRNGKey], float]]] = None

    def setup(self):
        self._f = self.param("f", self.f) if isinstance(self.f, Callable) else self.f
        self._n = self.param("n", self.n) if isinstance(self.n, Callable) else self.n
        self._NA = (
            self.param("NA", self.NA) if isinstance(self.NA, Callable) else self.NA
        )

    def __call__(self, field: Field) -> Field:
        return cf.thin_lens(field, self._f, self._n, self._NA)


class FFLens(nn.Module):
    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Optional[Union[float, Callable[[PRNGKey], float]]] = None

    def setup(self):
        self._f = self.param("f", self.f) if isinstance(self.f, Callable) else self.f
        self._n = self.param("n", self.n) if isinstance(self.n, Callable) else self.n
        self._NA = (
            self.param("NA", self.NA) if isinstance(self.NA, Callable) else self.NA
        )

    def __call__(self, field: Field) -> Field:
        return cf.ff_lens(field, self._f, self._n, self._NA)


class DFLens(nn.Module):
    d: Union[float, Callable[[PRNGKey], float]]
    f: Union[float, Callable[[PRNGKey], float]]
    n: Union[float, Callable[[PRNGKey], float]]
    NA: Optional[Union[float, Callable[[PRNGKey], float]]] = None

    def setup(self):
        self._d = self.param("d", self.d) if isinstance(self.d, Callable) else self.d
        self._f = self.param("f", self.f) if isinstance(self.f, Callable) else self.f
        self._n = self.param("n", self.n) if isinstance(self.n, Callable) else self.n
        self._NA = (
            self.param("NA", self.NA) if isinstance(self.NA, Callable) else self.NA
        )

    def __call__(self, field: Field) -> Field:
        return cf.df_lens(field, self._d, self._f, self._n, self._NA)
