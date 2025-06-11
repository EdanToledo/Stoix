from typing import NamedTuple

from chex import Array

from stoix.base_types import Action, Done, LogProb, Reward, Truncated


class ImpalaTransition(NamedTuple):
    done: Done
    truncated: Truncated
    action: Action
    reward: Reward
    log_prob: LogProb
    obs: Array
