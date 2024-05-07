from typing import Dict, Optional, Union

import chex
import optax
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, LogEnvState, OnlineAndTarget, Truncated


class SequenceStep(NamedTuple):
    obs: chex.ArrayTree
    action: Action
    reward: chex.Array
    done: Done
    truncated: Truncated
    log_prob: chex.Array
    info: Dict


class DualParams(NamedTuple):
    log_temperature: chex.Array
    log_alpha_mean: chex.Array
    log_alpha_stddev: chex.Array
    log_penalty_temperature: Optional[chex.Array] = None


class CategoricalDualParams(NamedTuple):
    log_temperature: chex.Array
    log_alpha: chex.Array


class MPOParams(NamedTuple):
    actor_params: OnlineAndTarget
    q_params: OnlineAndTarget
    dual_params: Union[DualParams, CategoricalDualParams]


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


class CategoricalMPOStats(NamedTuple):
    dual_alpha: float
    dual_temperature: float

    loss_e_step: float
    loss_m_step: float
    loss_dual: float

    loss_policy: float
    loss_alpha: float
    loss_temperature: float

    kl_q_rel: float
    kl_mean_rel: float

    q_min: float
    q_max: float

    entropy_online: float
    entropy_target: float


class VMPOParams(NamedTuple):
    actor_params: OnlineAndTarget
    critic_params: FrozenDict
    dual_params: Union[DualParams, CategoricalDualParams]


class VMPOOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    dual_opt_state: optax.OptState


class VMPOLearnerState(NamedTuple):
    params: VMPOParams
    opt_states: VMPOOptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    learner_step_count: int
