import functools
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from stoix.networks.utils import parse_rnn_cell


class ScannedRNN(nn.Module):
    hidden_state_dim: int
    cell_type: str

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, rnn_state: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        ins, resets = x
        hidden_state_reset_fn = lambda reset_state, current_state: jnp.where(
            resets[:, np.newaxis],
            reset_state,
            current_state,
        )
        rnn_state = jax.tree_util.tree_map(
            hidden_state_reset_fn,
            self.initialize_carry(ins.shape[0]),
            rnn_state,
        )
        new_rnn_state, y = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)(
            rnn_state, ins
        )
        return new_rnn_state, y

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, self.hidden_state_dim))
