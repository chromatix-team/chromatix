import flax.linen as nn
from typing import Callable, Optional, Union
from chex import PRNGKey
from chromatix import Field
import chromatix.functional as cf


class LinearPolarizer(nn.Module):
    """
    Applies a thin polarizer placed after ``Field``.
    This element returns the ``Field`` directly after the polarizer.

    This element can be placed after any element that returns a ``Field`` or
    before any element that accepts a ``Field``.

    Attributes:
        pangle: linear polarizer angle
    """
