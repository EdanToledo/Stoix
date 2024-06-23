from typing import Dict

import chex
from flashbax.buffers.trajectory_buffer import BufferState
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from stoix.base_types import (
    ActorCriticOptStates,
    ActorCriticParams,
    Done,
    LogEnvState,
    Truncated,
)


class AWRLearnerState(NamedTuple):
    params: ActorCriticParams
    opt_states: ActorCriticOptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class SequenceStep(NamedTuple):
    obs: chex.ArrayTree
    action: chex.Array
    reward: chex.Array
    done: Done
    truncated: Truncated
    info: Dict
