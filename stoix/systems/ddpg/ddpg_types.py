from typing import Callable

import chex
import optax
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from stoix.base_types import Action, LogEnvState, Observation, Value
from stoix.systems.q_learning.types import QsAndTarget

ContinuousQApply = Callable[[FrozenDict, Observation, Action], Value]


class ActorAndTarget(NamedTuple):
    online: FrozenDict
    target: FrozenDict


class DDPGParams(NamedTuple):
    actor_params: ActorAndTarget
    q_params: QsAndTarget


class DDPGOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState


class DDPGLearnerState(NamedTuple):
    params: DDPGParams
    opt_states: DDPGOptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
