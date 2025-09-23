from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import chex
from distrax import DistributionLike
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from optax import OptState
from stoa import TimeStep, WrapperState
from typing_extensions import NamedTuple, Protocol, TypeAlias, runtime_checkable

from stoix.utils.running_statistics import RunningStatisticsState

Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
Truncated: TypeAlias = chex.Array
First: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array
LogProb: TypeAlias = chex.Array
Reward: TypeAlias = chex.Array
# Can't know the exact type of State.
State: TypeAlias = Any
Parameters: TypeAlias = Any
OptStates: TypeAlias = Any
HiddenStates: TypeAlias = Any
Metrics: TypeAlias = chex.ArrayTree


EvalResetFn = Callable[[chex.PRNGKey, int], Tuple[State, TimeStep]]


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


class OnPolicyLearnerState(NamedTuple):
    """State of the learner. Used for on-policy learners."""

    params: Parameters
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep


class RNNLearnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures."""

    params: Parameters
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep
    done: Done
    truncated: Truncated
    hstates: HiddenStates


class OffPolicyLearnerState(NamedTuple):
    params: Parameters
    opt_states: OptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep


class RNNOffPolicyLearnerState(NamedTuple):
    params: Parameters
    opt_states: OptStates
    buffer_state: BufferState
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep
    dones: Done
    truncated: Truncated
    hstates: HiddenStates


class OnlineAndTarget(NamedTuple):
    online: FrozenDict
    target: FrozenDict


StoixState = TypeVar(
    "StoixState",
)
StoixTransition = TypeVar(
    "StoixTransition",
)


class SebulbaExperimentOutput(NamedTuple, Generic[StoixState]):
    """Experiment output."""

    learner_state: StoixState
    train_metrics: Dict[str, chex.Array]


class AnakinExperimentOutput(NamedTuple, Generic[StoixState]):
    """Experiment output."""

    learner_state: StoixState
    episode_metrics: Dict[str, chex.Array]
    train_metrics: Dict[str, chex.Array]


class EvaluationOutput(NamedTuple, Generic[StoixState]):
    """Evaluation output."""

    learner_state: StoixState
    episode_metrics: Dict[str, chex.Array]


RNNObservation: TypeAlias = Tuple[Observation, Done]
LearnerFn = Callable[[StoixState], AnakinExperimentOutput[StoixState]]
SebulbaLearnerFn = Callable[
    [StoixState, List[StoixTransition]], SebulbaExperimentOutput[StoixState]
]
SebulbaEvalFn = Callable[[FrozenDict, chex.PRNGKey], Dict[str, chex.Array]]

ActorApply = Callable[..., DistributionLike]

ActFn = Callable[[FrozenDict, Observation, chex.PRNGKey], chex.Array]
CriticApply = Callable[[FrozenDict, Observation], Value]
DistributionCriticApply = Callable[[FrozenDict, Observation], DistributionLike]
ContinuousQApply = Callable[[FrozenDict, Observation, Action], Value]
ActorCriticApply = Callable[[FrozenDict, Observation], Tuple[DistributionLike, Value]]
RecActorApply = Callable[
    [FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, DistributionLike]
]
RecActFn = Callable[
    [FrozenDict, HiddenState, RNNObservation, chex.PRNGKey], Tuple[HiddenState, chex.Array]
]
RecCriticApply = Callable[[FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, Value]]


@runtime_checkable
class EvalFn(Protocol[StoixState]):
    """Evaluator function protocol that allows for optional running_statistics parameter."""

    def __call__(
        self,
        trained_params: FrozenDict,
        key: chex.PRNGKey,
        running_statistics: Optional[RunningStatisticsState] = None,
    ) -> EvaluationOutput[StoixState]:
        ...
