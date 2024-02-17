from typing import Callable, Dict

import chex
from flax import linen as nn


def parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "silu": nn.silu,
        "elu": nn.elu,
        "gelu": nn.gelu,
    }
    return activation_fns[activation_fn_name]
