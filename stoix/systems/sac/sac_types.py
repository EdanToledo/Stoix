from typing import Callable

import chex
import optax
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from stoix.base_types import Action, LogEnvState, Observation, Value
from stoix.systems.q_learning.dqn_types import QsAndTarget

ContinuousQApply = Callable[[FrozenDict, Observation, Action], Value]


class SACParams(NamedTuple):
    actor_params: FrozenDict
    q_params: QsAndTarget
    log_alpha: chex.Array


class SACOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState
    alpha_opt_state: optax.OptState


class SACLearnerState(NamedTuple):
    params: SACParams
    opt_states: SACOptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
