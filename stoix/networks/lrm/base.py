from typing import Tuple, TypeAlias

import chex
import flax.linen as nn

RecurrentState: TypeAlias = chex.Array
Reset: TypeAlias = chex.Array
Timestep: TypeAlias = chex.Array
InputEmbedding: TypeAlias = chex.Array
Inputs: TypeAlias = Tuple[InputEmbedding, Reset]
ScanInput: TypeAlias = chex.Array


class LRMCellBase(nn.Module):
    def __call__(
        self, recurrent_state: RecurrentState, inputs: Inputs
    ) -> Tuple[RecurrentState, chex.Array]:
        raise NotImplementedError

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> RecurrentState:
        raise NotImplementedError
