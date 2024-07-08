from typing import Any, Dict, List, Tuple

import chex
import flax.linen as nn

from stoix.networks.lrm.base import Inputs, LRMCellBase, RecurrentState
from stoix.networks.lrm.utils import parse_lrm_cell


class StackedLRM(nn.Module):
    lrm_cell_type: LRMCellBase
    cell_kwargs: Dict[str, Any]
    num_cells: int

    def setup(self) -> None:
        """Set up the LRM cells for the stacked LRM."""

        cell_cls = parse_lrm_cell(self.lrm_cell_type)
        self.cells = [cell_cls(**self.cell_kwargs) for _ in range(self.num_cells)]

    @nn.compact
    def __call__(
        self, all_states: List[RecurrentState], inputs: Inputs
    ) -> Tuple[RecurrentState, chex.Array]:
        # Ensure all_states is a list
        if not isinstance(all_states, list):
            all_states = [all_states]

        assert len(all_states) == len(
            self.cells
        ), f"Expected {len(self.cells)} states, got {len(all_states)}"
        x, starts = inputs
        new_states = []
        for cell, mem_state in zip(self.cells, all_states):
            new_mem_state, x = cell(mem_state, (x, starts))
            new_states.append(new_mem_state)

        return new_states, x

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> List[RecurrentState]:
        cell_cls = parse_lrm_cell(self.lrm_cell_type)
        cells = [cell_cls(**self.cell_kwargs) for _ in range(self.num_cells)]
        return [cell.initialize_carry(batch_size) for cell in cells]
