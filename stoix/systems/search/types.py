from typing import Callable, Dict, Tuple

import chex
import mctx
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from stoix.systems.ppo.types import ActorCriticParams
from stoix.types import Action, Done, Observation, Value

SearchApply = Callable[[FrozenDict, chex.PRNGKey, mctx.RootFnOutput], mctx.PolicyOutput]
RootFnApply = Callable[[FrozenDict, Observation, chex.ArrayTree], mctx.RootFnOutput]
EnvironmentStep = Callable[[chex.ArrayTree, Action], Tuple[chex.ArrayTree, TimeStep]]

RepresentationApply = Callable[[FrozenDict, Observation], chex.Array]
DynamicsApply = Callable[[FrozenDict, chex.Array, chex.Array], Tuple[chex.Array, chex.Array]]


class AZTransition(NamedTuple):
    """Transition tuple for AZ."""

    done: Done
    action: Action
    value: Value
    reward: chex.Array
    search_value: Value
    search_policy: chex.Array
    obs: chex.Array
    info: Dict


class WorldModelParams(NamedTuple):
    representation_params: FrozenDict
    dynamics_params: FrozenDict


class MZParams(NamedTuple):
    prediction_params: ActorCriticParams
    world_model_params: WorldModelParams


class MZOptStates(NamedTuple):
    actor_opt_state: chex.ArrayTree
    critic_opt_state: chex.ArrayTree
    world_model_opt_state: chex.ArrayTree


class MZLearnerState(NamedTuple):
    params: MZParams
    opt_states: MZOptStates
    buffer_state: chex.ArrayTree
    key: chex.PRNGKey
    env_state: TimeStep
    timestep: TimeStep
