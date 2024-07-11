import optax
from typing_extensions import NamedTuple

from stoix.base_types import OnlineAndTarget


class DDPGParams(NamedTuple):
    actor_params: OnlineAndTarget
    q_params: OnlineAndTarget


class DDPGOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState
