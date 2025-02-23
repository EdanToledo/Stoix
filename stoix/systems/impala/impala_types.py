from typing import Any, Dict, NamedTuple, Union

import jax
import numpy as np

from stoix.base_types import ActorCriticHiddenStates, Observation

Array = Union[np.ndarray, jax.Array]
LogProbArray = Array  # Array of log probabilities
ValueArray = Array  # Array of value estimates
ActionArray = Array  # Array of actions
RewardArray = Array  # Array of rewards
MetricsDict = Dict[str, Any]  # Dictionary of metrics


class ImpalaTransition(NamedTuple):
    done: Array
    truncated: Array
    action: ActionArray
    value: ValueArray
    reward: RewardArray
    log_prob: LogProbArray
    obs: Observation
    metrics: MetricsDict


class RNNImpalaTransition(NamedTuple):
    """Transition for recurrent IMPALA that includes hidden states."""

    done: Array
    truncated: Array
    action: ActionArray
    value: ValueArray
    reward: RewardArray
    log_prob: LogProbArray
    obs: Observation
    metrics: MetricsDict
    hstates: ActorCriticHiddenStates  # Hidden states for both actor and critic
    reset_hidden_state: Array  # Add this new field
