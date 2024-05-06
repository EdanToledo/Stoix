from typing import Dict

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, ActorCriticHiddenStates, Done, Truncated, Value


class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: Done
    truncated: Truncated
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict


class RNNPPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: Done
    truncated: Truncated
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    hstates: ActorCriticHiddenStates
    info: Dict
