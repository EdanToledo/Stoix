from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

import chex
from distrax import DistributionLike
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from optax import OptState
from typing_extensions import NamedTuple, TypeAlias

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
Truncated: TypeAlias = chex.Array
First: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array
# Can't know the exact type of State.
State: TypeAlias = Any
Parameters: TypeAlias = Any
OptStates: TypeAlias = Any
HiddenStates: TypeAlias = Any


class Observation(NamedTuple):
    """The observation that the agent sees.
    agent_view: the agent's view of the environment.
    action_mask: boolean array specifying which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_view: chex.Array  # (num_obs_features,)
    action_mask: chex.Array  # (num_actions,)
    step_count: Optional[chex.Array] = None  # (,)


class ObservationGlobalState(NamedTuple):
    """The observation seen by agents in centralised systems.
    Extends `Observation` by adding a `global_state` attribute for centralised training.
    global_state: The global state of the environment, often a concatenation of agents' views.
    """

    agent_view: chex.Array
    action_mask: chex.Array
    global_state: chex.Array
    step_count: chex.Array


@dataclass
class LogEnvState:
    """State of the `LogWrapper`."""

    env_state: State
    episode_returns: chex.Numeric
    episode_lengths: chex.Numeric
    # Information about the episode return and length for logging purposes.
    episode_return_info: chex.Numeric
    episode_length_info: chex.Numeric


class EvalState(NamedTuple):
    """State of the evaluator."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    step_count: chex.Array
    episode_return: chex.Array


class RNNEvalState(NamedTuple):
    """State of the evaluator for recurrent architectures."""

    key: chex.PRNGKey
    env_state: State
    timestep: TimeStep
    dones: Done
    hstate: HiddenState
    step_count: chex.Array
    episode_return: chex.Array


class ActorCriticParams(NamedTuple):
    """Parameters of an actor critic network."""

    actor_params: FrozenDict
    critic_params: FrozenDict


class ActorCriticOptStates(NamedTuple):
    """OptStates of actor critic learner."""

    actor_opt_state: OptState
    critic_opt_state: OptState


class ActorCriticHiddenStates(NamedTuple):
    """Hidden states for an actor critic learner."""

    policy_hidden_state: HiddenState
    critic_hidden_state: HiddenState


class CoreLearnerState(NamedTuple):
    """Base state of the learner. Can be used for both on-policy and off-policy learners.
    Mainly used for sebulba systems since we dont store env state."""

    params: Parameters
    opt_states: OptStates
    key: chex.PRNGKey
    timestep: TimeStep


class OnPolicyLearnerState(NamedTuple):
    """State of the learner. Used for on-policy learners."""

    params: Parameters
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class RNNLearnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures."""

    params: Parameters
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep
    done: Done
    truncated: Truncated
    hstates: HiddenStates


class OffPolicyLearnerState(NamedTuple):
    params: Parameters
    opt_states: OptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: LogEnvState
    timestep: TimeStep


class OnlineAndTarget(NamedTuple):
    online: FrozenDict
    target: FrozenDict


StoixState = TypeVar(
    "StoixState",
)
StoixTransition = TypeVar(
    "StoixTransition",
)


class ExperimentOutput(NamedTuple, Generic[StoixState]):
    """Experiment output."""

    learner_state: StoixState
    episode_metrics: Dict[str, chex.Array]
    train_metrics: Dict[str, chex.Array]


RNNObservation: TypeAlias = Tuple[Observation, Done]
LearnerFn = Callable[[StoixState], ExperimentOutput[StoixState]]
SebulbaLearnerFn = Callable[[StoixState, StoixTransition], ExperimentOutput[StoixState]]
EvalFn = Callable[[FrozenDict, chex.PRNGKey], ExperimentOutput[StoixState]]

ActorApply = Callable[..., DistributionLike]

ActFn = Callable[[FrozenDict, Observation, chex.PRNGKey], chex.Array]
CriticApply = Callable[[FrozenDict, Observation], Value]
DistributionCriticApply = Callable[[FrozenDict, Observation], DistributionLike]
ContinuousQApply = Callable[[FrozenDict, Observation, Action], Value]

RecActorApply = Callable[
    [FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, DistributionLike]
]
RecActFn = Callable[
    [FrozenDict, HiddenState, RNNObservation, chex.PRNGKey], Tuple[HiddenState, chex.Array]
]
RecCriticApply = Callable[[FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, Value]]


EnvFactory = Callable[[int], Any]
