from typing import Callable, Dict, Tuple, Union

import chex
import mctx
from distrax import DistributionLike
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax import OptState
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, Observation, Value
from stoix.systems.ppo.ppo_types import ActorCriticParams

SearchApply = Callable[[FrozenDict, chex.PRNGKey, mctx.RootFnOutput], mctx.PolicyOutput]
RootFnApply = Callable[[FrozenDict, Observation, chex.ArrayTree, chex.PRNGKey], mctx.RootFnOutput]
EnvironmentStep = Callable[[chex.ArrayTree, Action], Tuple[chex.ArrayTree, TimeStep]]

RepresentationApply = Callable[[FrozenDict, Observation], chex.Array]
DynamicsApply = Callable[[FrozenDict, chex.Array, chex.Array], Tuple[chex.Array, DistributionLike]]


class ExItTransition(NamedTuple):
    done: Done
    action: Action
    reward: chex.Array
    search_value: Value
    search_policy: chex.Array
    obs: chex.Array
    info: Dict


class SampledExItTransition(NamedTuple):
    done: chex.Array
    action: Action
    sampled_actions: chex.Array
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


class ZLearnerState(NamedTuple):
    params: Union[MZParams, ActorCriticParams]
    opt_states: OptState
    buffer_state: chex.ArrayTree
    key: chex.PRNGKey
    env_state: TimeStep
    timestep: TimeStep
