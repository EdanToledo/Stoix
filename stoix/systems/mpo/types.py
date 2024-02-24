from typing import Dict, Optional, Union

import chex
import optax
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from stoix.systems.q_learning.types import QsAndTarget
from stoix.types import LogEnvState


class SequenceStep(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    log_prob: chex.Array
    info: Dict


class ActorAndTarget(NamedTuple):
    online: FrozenDict
    target: FrozenDict


class DualParams(NamedTuple):
    log_temperature: chex.Array
    log_alpha_mean: chex.Array
    log_alpha_stddev: chex.Array
    log_penalty_temperature: Optional[chex.Array] = None


class MPOParams(NamedTuple):
    actor_params: FrozenDict
    q_params: QsAndTarget
    dual_params: DualParams


class MPOOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState
    dual_opt_state: optax.OptState


class MPOLearnerState(NamedTuple):
    params: MPOParams
    opt_states: MPOOptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class MPOStats(NamedTuple):
    dual_alpha_mean: Union[float, chex.Array]
    dual_alpha_stddev: Union[float, chex.Array]
    dual_temperature: Union[float, chex.Array]

    loss_policy: Union[float, chex.Array]
    loss_alpha: Union[float, chex.Array]
    loss_temperature: Union[float, chex.Array]
    kl_q_rel: Union[float, chex.Array]

    kl_mean_rel: Union[float, chex.Array]
    kl_stddev_rel: Union[float, chex.Array]

    q_min: Union[float, chex.Array]
    q_max: Union[float, chex.Array]

    pi_stddev_min: Union[float, chex.Array]
    pi_stddev_max: Union[float, chex.Array]
    pi_stddev_cond: Union[float, chex.Array]

    penalty_kl_q_rel: Optional[float] = None
