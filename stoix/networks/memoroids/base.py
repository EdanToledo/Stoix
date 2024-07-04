from typing import List, Optional, Tuple

import chex
from flax import linen as nn
from jax import numpy as jnp

RecurrentState = chex.Array
Reset = chex.Array
Timestep = chex.Array
InputEmbedding = chex.Array
Inputs = Tuple[InputEmbedding, Reset]
ScanInput = chex.Array


class MemoroidCellBase(nn.Module):
    """Memoroid cell base class."""

    def map_to_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> RecurrentState:
        raise NotImplementedError

    def map_from_h(self, recurrent_state: RecurrentState, x: InputEmbedding) -> RecurrentState:
        raise NotImplementedError

    def scan(self, x: InputEmbedding, state: RecurrentState, start: Reset) -> RecurrentState:
        raise NotImplementedError

    @nn.nowrap
    def initialize_carry(
        self, batch_size: Optional[int] = None, rng: Optional[chex.PRNGKey] = None
    ) -> RecurrentState:
        """Initialize the Memoroid cell carry.

        Args:
            batch_size: the batch size of the carry.
            rng: random number generator passed to the init_fn.

        Returns:
        An initialized carry for the given Memoroid cell.
        """
        raise NotImplementedError

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the cell."""
        raise NotImplementedError


# class ScannedMemoroid(nn.Module):
#     cell: MemoroidCellBase

#     @nn.compact
#     def __call__(self, state: RecurrentState, inputs: Inputs) -> Tuple[RecurrentState, chex.Array]:

#         # Add a sequence dimension to the recurrent state.
#         state = jnp.expand_dims(state, 0)

#         # Unpack inputs
#         x, start = inputs

#         # Map the input embedding to the recurrent state space.
#         # This maps to the format required for the associative scan.
#         scan_input = self.cell.map_to_h(x)

#         # Update the recurrent state
#         state = self.cell.scan(scan_input, state, start)

#         # Map the recurrent state back to the output space
#         out = self.cell.map_from_h(state, x)

#         # Take the final state of the sequence.
#         final_state = state[-1:]

#         # Remove the sequence dimemnsion from the final state.
#         final_state = jnp.squeeze(final_state, 0)

#         return final_state, out

#     @nn.nowrap
#     def initialize_carry(self, batch_size: int) -> RecurrentState:
#         return self.cell.initialize_carry(batch_size)


# class StackedMemoroid(nn.Module):
#     cells: Tuple[ScannedMemoroid]

#     @nn.compact
#     def __call__(
#         self, all_states: List[RecurrentState], inputs: Inputs
#     ) -> Tuple[RecurrentState, chex.Array]:
#         # Ensure all_states is a list
#         if not isinstance(all_states, list):
#             all_states = [all_states]

#         assert len(all_states) == len(
#             self.cells
#         ), f"Expected {len(self.cells)} states, got {len(all_states)}"
#         x, starts = inputs
#         new_states = []
#         for cell, mem_state in zip(self.cells, all_states):
#             new_mem_state, x = cell(mem_state, (x, starts))
#             new_states.append(new_mem_state)

#         return new_states, x

#     @nn.nowrap
#     def initialize_carry(self, batch_size: int) -> List[RecurrentState]:
#         return [cell.initialize_carry(batch_size) for cell in self.cells]
