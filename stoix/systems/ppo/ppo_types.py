from typing import Dict

import chex
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax._src.base import OptState
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done, HiddenState, LogEnvState, Truncated, Value


class ActorCriticParams(NamedTuple):
    """Parameters of an actor critic network."""

    actor_params: FrozenDict
    critic_params: FrozenDict


class ActorCriticOptStates(NamedTuple):
    """OptStates of actor critic learner."""

    actor_opt_state: OptState
    critic_opt_state: OptState


class HiddenStates(NamedTuple):
    """Hidden states for an actor critic learner."""

    policy_hidden_state: HiddenState
    critic_hidden_state: HiddenState


class LearnerState(NamedTuple):
    """State of the learner."""

    params: ActorCriticParams
    opt_states: ActorCriticOptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class RNNLearnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures."""

    params: ActorCriticParams
    opt_states: ActorCriticOptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    dones: Done
    hstates: HiddenStates


class PPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: Done
    truncated: Truncated
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict


class RNNPPOTransition(NamedTuple):
    """Transition tuple for PPO."""

    done: Done
    truncated: Truncated
    action: Action
    value: Value
    reward: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    hstates: HiddenStates
    info: Dict
