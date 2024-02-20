from typing import Callable, Dict, Tuple

import chex
import mctx
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from typing_extensions import NamedTuple

from stoix.types import Action, Done, Observation, Value

SearchApply = Callable[[FrozenDict, chex.PRNGKey, mctx.RootFnOutput], mctx.PolicyOutput]
RootFnApply = Callable[[FrozenDict, Observation, chex.ArrayTree], mctx.RootFnOutput]
EnvironmentStep = Callable[[chex.ArrayTree, Action], Tuple[chex.ArrayTree, TimeStep]]


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
