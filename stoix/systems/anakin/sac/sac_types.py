import chex
import optax
from flax.core.frozen_dict import FrozenDict
from typing_extensions import NamedTuple

from stoix.base_types import OnlineAndTarget


class SACParams(NamedTuple):
    actor_params: FrozenDict
    q_params: OnlineAndTarget
    log_alpha: chex.Array


class SACOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState
    alpha_opt_state: optax.OptState
