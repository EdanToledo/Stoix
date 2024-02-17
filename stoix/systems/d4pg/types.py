from typing import Callable

import chex
import optax
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from stoix.systems.q_learning.types import QsAndTarget
from stoix.types import Action, LogEnvState, Observation, Value

ContinuousQApply = Callable[[FrozenDict, Observation, Action], Value]


class ActorAndTarget(NamedTuple):
    online: FrozenDict
    target: FrozenDict


class D4PGParams(NamedTuple):
    actor_params: ActorAndTarget
    q_params: QsAndTarget


class D4PGOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    q_opt_state: optax.OptState


class D4PGLearnerState(NamedTuple):
    params: D4PGParams
    opt_states: D4PGOptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
