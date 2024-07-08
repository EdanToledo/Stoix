from typing import Any, Dict

from stoix.networks.lrm.ffm import FFMCell
from stoix.networks.lrm.lru import LRUCell
from stoix.networks.lrm.s5 import S5Cell


def parse_lrm_cell(lrm_cell_name: str) -> Any:
    """Parse a linear recurrent model layer."""
    lrm_cells: Dict[str, Any] = {
        "s5": S5Cell,
        "ffm": FFMCell,
        "lru": LRUCell,
    }
    return lrm_cells[lrm_cell_name]
