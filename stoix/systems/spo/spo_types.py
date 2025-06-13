from typing import Callable, Dict, NamedTuple, Tuple, TypeAlias, Union

import chex
import optax
from flax.core.frozen_dict import FrozenDict

from stoix.base_types import (
    Action,
    Done,
    Observation,
    OnlineAndTarget,
    Truncated,
    Value,
)
from stoix.systems.mpo.mpo_types import CategoricalDualParams, DualParams

_SPO_FLOAT_EPSILON = 1e-8


class SPOParams(NamedTuple):
    actor_params: OnlineAndTarget
    critic_params: OnlineAndTarget
    dual_params: Union[CategoricalDualParams, DualParams]


class SPOOptStates(NamedTuple):
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    dual_opt_state: optax.OptState


class SPOTransition(NamedTuple):
    done: Done
    truncated: Truncated
    action: Action
    sampled_actions: chex.Array
    sampled_actions_weights: chex.Array
    reward: chex.Array
    search_value: Value
    obs: chex.Array
    bootstrap_obs: chex.Array
    info: Dict
    sampled_advantages: chex.Array


class SPORootFnOutput(NamedTuple):
    particle_logits: chex.Array
    particle_actions: chex.Array
    particle_env_states: chex.ArrayTree
    particle_values: chex.Array


class SPORecurrentFnOutput(NamedTuple):
    reward: chex.Array
    discount: chex.Array
    prior_logits: chex.Array
    value: chex.Array
    next_sampled_action: chex.Array


class SPOOutput(NamedTuple):
    action: Action
    sampled_action_weights: chex.Array
    sampled_actions: chex.Array
    value: chex.Array
    sampled_advantages: chex.Array
    rollout_metrics: Dict


StateEmbedding: TypeAlias = chex.ArrayTree
SPOApply = Callable[[FrozenDict, chex.PRNGKey, SPORootFnOutput], SPOOutput]
SPORootFnApply = Callable[[FrozenDict, Observation, chex.ArrayTree, chex.PRNGKey], SPORootFnOutput]
SPORecurrentFn = Callable[
    [SPOParams, chex.PRNGKey, Action, StateEmbedding], Tuple[SPORecurrentFnOutput, StateEmbedding]
]
