from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import HiddenState


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    info: Dict


class RNNTransition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    reset_hidden_state: chex.Array
    done: chex.Array
    truncated: chex.Array
    info: Dict
    hstate: HiddenState
