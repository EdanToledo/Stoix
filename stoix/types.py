from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, Tuple, TypeVar

import chex
from distrax import DistributionLike
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple, TypeAlias

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


Action: TypeAlias = chex.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array
First: TypeAlias = chex.Array
HiddenState: TypeAlias = chex.Array
# Can't know the exact type of State.
State: TypeAlias = Any


class Observation(NamedTuple):
    """The observation that the agent sees.
    agent_view: the agent's view of the environment.
    action_mask: boolean array specifying which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agent_view: chex.Array  # (num_obs_features,)
    action_mask: chex.Array  # (num_actions,)
    step_count: chex.Array  # (,)


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
    dones: chex.Array
    hstate: HiddenState
    step_count: chex.Array
    episode_return: chex.Array


StoixState = TypeVar(
    "StoixState",
)


class ExperimentOutput(NamedTuple, Generic[StoixState]):
    """Experiment output."""

    learner_state: StoixState
    episode_metrics: Dict[str, chex.Array]
    train_metrics: Dict[str, chex.Array]


RNNObservation: TypeAlias = Tuple[Observation, Done]
LearnerFn = Callable[[StoixState], ExperimentOutput[StoixState]]
EvalFn = Callable[[FrozenDict, chex.PRNGKey], ExperimentOutput[StoixState]]

ActorApply = Callable[[FrozenDict, Observation], DistributionLike]
CriticApply = Callable[[FrozenDict, Observation], Value]
RecActorApply = Callable[
    [FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, DistributionLike]
]
RecCriticApply = Callable[[FrozenDict, HiddenState, RNNObservation], Tuple[HiddenState, Value]]
