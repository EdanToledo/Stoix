from typing import Dict, Union

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
