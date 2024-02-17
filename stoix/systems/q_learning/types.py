from typing import Dict

import chex
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from stoix.types import LogEnvState


class QsAndTarget(NamedTuple):
    online: FrozenDict
    target: FrozenDict


class Transition(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    info: Dict


class DQNLearnerState(NamedTuple):
    params: QsAndTarget
    opt_states: FrozenDict
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
