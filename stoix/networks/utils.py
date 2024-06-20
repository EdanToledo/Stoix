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
        "sigmoid": nn.sigmoid,
        "softplus": nn.softplus,
        "swish": nn.swish,
        "identity": lambda x: x,
        "none": lambda x: x,
        "normalise": nn.standardize,
        "softmax": nn.softmax,
        "log_softmax": nn.log_softmax,
        "log_sigmoid": nn.log_sigmoid,
    }
    return activation_fns[activation_fn_name]


def parse_rnn_cell(rnn_cell_name: str) -> nn.RNNCellBase:
    """Get the rnn cell."""
    rnn_cells: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "lstm": nn.LSTMCell,
        "optimised_lstm": nn.OptimizedLSTMCell,
        "gru": nn.GRUCell,
        "mgu": nn.MGUCell,
        "simple": nn.SimpleCell,
    }
    return rnn_cells[rnn_cell_name]
