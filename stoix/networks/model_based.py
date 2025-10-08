from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from stoix.base_types import Observation
from stoix.networks.inputs import ArrayInput
from stoix.networks.layers import StackedRNN
from stoix.networks.utils import parse_activation_fn, parse_rnn_cell


class RewardBasedWorldModel(nn.Module):
    """
    A RewardBasedWorldModel that supports both MuZero and EfficientZero-style dynamics.

    - For MuZero-type models, reward is predicted from the dynamics RNN output.
    - For EfficientZero-type models (`use_reward_rnn=True`), a separate RNN is used for
      reward prediction, allowing its state to be reset independently.
    """

    obs_encoder: nn.Module
    reward_torso: nn.Module
    reward_head: nn.Module
    rnn_size: int
    action_dim: int
    num_stacked_rnn_layers: int
    use_reward_rnn: bool = False
    normalize_encoded_state: bool = True
    rnn_cell_type: str = "lstm"
    recurrent_activation: str = "tanh"
    nonlinear_to_encoded_state: bool = False
    embed_actions: bool = True
    observation_input_layer: nn.Module = ArrayInput()

    def setup(self) -> None:
        """Initialise the network components."""
        self._to_dynamics_state = nn.Dense(self.encoded_state_size)

        if self.embed_actions:
            self._action_embeddings = nn.Dense(self.encoded_state_size)

        rnn_cell_cls = parse_rnn_cell(self.rnn_cell_type)

        self._dynamics_core = StackedRNN(
            self.rnn_size, rnn_cell_cls, self.num_stacked_rnn_layers, self.recurrent_activation
        )

        if self.use_reward_rnn:
            self._reward_core = StackedRNN(
                self.rnn_size, rnn_cell_cls, self.num_stacked_rnn_layers, self.recurrent_activation
            )

    @cached_property
    def encoded_state_size(self) -> int:
        """Calculate the size of the flattened RNN state."""
        if self.rnn_cell_type in ("gru", "simple"):
            return self.rnn_size * self.num_stacked_rnn_layers
        elif self.rnn_cell_type in ("lstm", "optimised_lstm"):
            return self.rnn_size * self.num_stacked_rnn_layers * 2
        raise ValueError(f"Unknown rnn_cell_type: {self.rnn_cell_type}")

    def _rnn_to_flat(self, state: List[chex.ArrayTree]) -> chex.Array:
        """Maps a list of RNN states to a flat vector."""
        states = []
        for cell_state in state:
            if not isinstance(cell_state, (list, tuple)):
                cell_state = (cell_state,)
            states.extend(cell_state)
        return jnp.concatenate(states, axis=-1)

    def _flat_to_rnn(self, state: chex.Array) -> List[chex.ArrayTree]:
        """Maps a flat vector back to a structured RNN state."""
        tensors = []
        cur_idx = 0
        for _ in range(self.num_stacked_rnn_layers):
            if self.rnn_cell_type in ("gru", "simple"):
                states = state[..., cur_idx : cur_idx + self.rnn_size]
                cur_idx += self.rnn_size
            elif self.rnn_cell_type in ("lstm", "optimised_lstm"):
                states = (
                    state[..., cur_idx : cur_idx + self.rnn_size],
                    state[..., cur_idx + self.rnn_size : cur_idx + 2 * self.rnn_size],
                )
                cur_idx += 2 * self.rnn_size
            tensors.append(states)
        chex.assert_equal(cur_idx, state.shape[-1])
        return tensors

    def _maybe_normalize_encoded_state(self, encoded_state: chex.Array) -> chex.Array:
        if self.normalize_encoded_state:
            max_encoded_state = jnp.max(encoded_state, axis=-1, keepdims=True)
            min_encoded_state = jnp.min(encoded_state, axis=-1, keepdims=True)
            encoded_state_range = max_encoded_state - min_encoded_state
            encoded_state = (encoded_state - min_encoded_state) / (
                encoded_state_range + 1e-8
            ) * 2.0 - 1.0
        return encoded_state

    def representation(self, observation: Observation) -> chex.Array:
        """
        The Representation Function (h).
        Encodes an observation into an initial hidden state for the dynamics model.
        """
        # Process observation with optional input layer
        observation = self.observation_input_layer(observation)
        # Encode observation
        encoded_observation = self.obs_encoder(observation)
        # Project to the size of the dynamics RNN state
        dynamics_state = self._to_dynamics_state(encoded_observation)
        # Optional nonlinearity
        if self.nonlinear_to_encoded_state:
            dynamics_state = parse_activation_fn(self.recurrent_activation)(dynamics_state)

        return dynamics_state

    def dynamics(
        self,
        dynamics_state: chex.Array,
        action: chex.Array,
        reward_rnn_state: Optional[chex.Array] = None,
    ) -> Union[Tuple[chex.Array, chex.Array], Tuple[chex.Array, chex.Array, chex.Array]]:
        """
        The Dynamics Function (g).
        Given a hidden state and action, predicts the next hidden state and reward.
        """
        # Embed action
        if self.embed_actions:
            if action.dtype == jnp.int32:
                action = jax.nn.one_hot(action, self.action_dim)
            embedded_action = self._action_embeddings(action)
        else:
            if action.dtype == jnp.int32:
                action = jax.nn.one_hot(action, self.action_dim)
            embedded_action = action

        # Normalize state before feeding to RNN
        dynamics_state = self._maybe_normalize_encoded_state(dynamics_state)

        # Run the dynamics RNN
        rnn_state = self._flat_to_rnn(dynamics_state)
        next_rnn_state, rnn_output = self._dynamics_core(rnn_state, embedded_action)
        next_dynamics_state_flat = self._rnn_to_flat(next_rnn_state)

        # Add residual connection
        next_dynamics_state = next_dynamics_state_flat + dynamics_state

        # Predict reward based on the configured mode
        if self.use_reward_rnn:
            reward_rnn_state_structured = self._flat_to_rnn(reward_rnn_state)
            next_reward_rnn_state_structured, reward_rnn_output = self._reward_core(
                reward_rnn_state_structured, rnn_output
            )
            reward = self.reward_head(self.reward_torso(reward_rnn_output))
            next_reward_rnn_state = self._rnn_to_flat(next_reward_rnn_state_structured)
            return next_dynamics_state, reward, next_reward_rnn_state
        else:
            reward = self.reward_head(self.reward_torso(rnn_output))
            return next_dynamics_state, reward

    def initial_state(self, batch_size: int) -> Dict[str, chex.Array]:
        """Returns a dictionary of initial zero states for all RNNs."""
        dynamics_state = jnp.zeros((batch_size, self.encoded_state_size))
        states = {"dynamics_state": dynamics_state}
        if self.use_reward_rnn:
            states["reward_rnn_state"] = jnp.zeros((batch_size, self.encoded_state_size))
        return states

    def __call__(
        self,
        observation: Observation,
        action: chex.Array,
    ) -> chex.Array:
        """Only used for network initialisation."""
        dynamics_state = self.representation(observation)
        return self.dynamics(
            dynamics_state, action, self.initial_state(action.shape[0]).get("reward_rnn_state")
        )
