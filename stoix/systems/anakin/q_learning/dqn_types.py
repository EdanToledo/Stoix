from typing import Dict

import chex
from typing_extensions import NamedTuple


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    info: Dict
