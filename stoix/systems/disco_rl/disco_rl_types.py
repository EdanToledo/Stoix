# stoix/systems/disco/disco_types.py
from typing import Dict

import chex
from stoa import TimeStep, WrapperState
from typing_extensions import NamedTuple

from disco_rl import types as disco_types
from stoix.base_types import ActorCriticOptStates, ActorCriticParams


class AgentOutput(NamedTuple):
    """Network output for the Disco103 agent. Contains the outputs required by the DiscoUpdateRule."""

    logits: chex.Array  # Policy
    q: chex.Array  # Categorical Value
    y: chex.Array  # Auxiliary head
    z: chex.Array  # Auxiliary head
    aux_pi: chex.Array  # Auxiliary policy head


class DiscoTransition(NamedTuple):
    """Transition structure for Disco103 rollouts."""

    done: chex.Array
    truncated: chex.Array
    action: chex.Array
    reward: chex.Array
    obs: chex.Array
    info: Dict
    agent_out: AgentOutput


class DiscoLearnerState(NamedTuple):
    """The complete state of the Disco learner."""

    params: ActorCriticParams
    opt_states: ActorCriticOptStates
    key: chex.PRNGKey
    env_state: WrapperState
    timestep: TimeStep
    # The fixed, pre-trained parameters for the meta-network (e.g., disco103).
    meta_params: disco_types.MetaParams
    # The evolving internal state of the meta-network (RNN state, EMAs, target_params).
    meta_state: disco_types.MetaState  # type: ignore
