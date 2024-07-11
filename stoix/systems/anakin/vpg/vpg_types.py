from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Value


class Transition(NamedTuple):
    done: Done
    action: Action
    value: Value
    reward: chex.Array
    obs: chex.Array
    info: Dict
