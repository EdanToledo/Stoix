from typing import Callable, Dict, Tuple, Union

import chex
import mctx
from distrax import DistributionLike
from flax.core.frozen_dict import FrozenDict
from optax import OptState
from stoa import TimeStep, WrapperState
from typing_extensions import NamedTuple

from stoix.base_types import Action, ActorCriticParams, AgentObservation, Done, Value

SearchApply = Callable[[FrozenDict, chex.PRNGKey, mctx.RootFnOutput], mctx.PolicyOutput]
RootFnApply = Callable[
    [FrozenDict, AgentObservation, chex.ArrayTree, chex.PRNGKey], mctx.RootFnOutput
]
EnvironmentStep = Callable[[chex.ArrayTree, Action], Tuple[chex.ArrayTree, TimeStep]]

RepresentationApply = Callable[[FrozenDict, AgentObservation], chex.Array]
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


class MZParams(NamedTuple):
    prediction_params: ActorCriticParams
    world_model_params: FrozenDict


class ZLearnerState(NamedTuple):
    params: Union[MZParams, ActorCriticParams]
    opt_states: OptState
    buffer_state: chex.ArrayTree
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep
