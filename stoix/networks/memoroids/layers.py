from typing import Any, Callable, Dict, List, Tuple

import chex
from flax import linen as nn

from stoix.networks.memoroids.base import Inputs, MemoroidCellBase, RecurrentState
from stoix.networks.memoroids.ffm import FFMCell
from stoix.networks.memoroids.lru import LRUCell
from stoix.networks.memoroids.s5 import S5Cell


def parse_lrm_cell(lrm_cell_name: str) -> MemoroidCellBase:
    """Get the lrm cell."""
    lrm_cells: Dict[str, MemoroidCellBase] = {
        "s5": S5Cell,
        "ffm": FFMCell,
        "lru": LRUCell,
    }
    return lrm_cells[lrm_cell_name]


class StackedMemoroid(nn.Module):
    lrm_cell_type: MemoroidCellBase
    cell_kwargs: Dict[str, Any]
    num_cells: int

    def setup(self) -> None:
        """Set up the Memoroid cells for the stacked Memoroid."""

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
