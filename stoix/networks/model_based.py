from functools import cached_property
from typing import List

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from stoix.base_types import Observation
from stoix.networks.inputs import ObservationInput
from stoix.networks.layers import StackedRNN
from stoix.networks.utils import parse_activation_fn, parse_rnn_cell


class RewardBasedWorldModel(nn.Module):
    obs_encoder: nn.Module
    reward_torso: nn.Module
    reward_head: nn.Module
    rnn_size: int
    action_dim: int
    num_stacked_rnn_layers: int
    normalize_hidden_state: bool = True
    rnn_cell_type: str = "lstm"
    recurrent_activation: str = "tanh"
    nonlinear_to_hidden: bool = False
    embed_actions: bool = True
    observation_input_layer: nn.Module = ObservationInput()

    def setup(self) -> None:
        self._to_hidden = nn.Dense(self.hidden_state_size)

        if self.embed_actions:
            self._action_embeddings = nn.Dense(self.hidden_state_size)

        rnn_cell_cls = parse_rnn_cell(self.rnn_cell_type)

        self._core = StackedRNN(
            self.rnn_size, rnn_cell_cls, self.num_stacked_rnn_layers, self.recurrent_activation
        )

    @cached_property
    def hidden_state_size(self) -> int:
        if self.rnn_cell_type in ("gru", "simple"):
            hidden_state_size = sum([self.rnn_size] * self.num_stacked_rnn_layers)
        elif self.rnn_cell_type in ("lstm", "optimised_lstm"):
            hidden_state_size = sum([self.rnn_size * self.num_stacked_rnn_layers]) * 2
        return hidden_state_size

    def _rnn_to_flat(self, state: List[chex.ArrayTree]) -> chex.Array:
        """Maps list of RNN states to flat vector."""
        states = []
        for cell_state in state:
            if not (isinstance(cell_state, list) or isinstance(cell_state, tuple)):
                # This is a GRU or SimpleRNNCell
                cell_state = (cell_state,)
            states.extend(cell_state)
        return jnp.concatenate(states, axis=-1)

    def _flat_to_rnn(self, state: chex.Array) -> List[chex.ArrayTree]:
        """Maps flat vector to RNN state."""
        tensors = []
        cur_idx = 0
        for _ in range(self.num_stacked_rnn_layers):
            if self.rnn_cell_type in ("gru", "simple"):
                states = state[Ellipsis, cur_idx : cur_idx + self.rnn_size]
                cur_idx += self.rnn_size
            elif self.rnn_cell_type in ("lstm", "optimised_lstm"):
                states = (
                    state[Ellipsis, cur_idx : cur_idx + self.rnn_size],
                    state[Ellipsis, cur_idx + self.rnn_size : cur_idx + 2 * self.rnn_size],
                )
                cur_idx += 2 * self.rnn_size
            tensors.append(states)
        assert cur_idx == state.shape[-1]
        return tensors

    def initial_state(self, batch_size: int) -> chex.Array:
        return jnp.zeros((batch_size, self.hidden_state_size))

    def _encode_observation(self, observation: Observation) -> chex.Array:
        observation = self.observation_input_layer(observation)
        return self.obs_encoder(observation)

    def initial_inference(self, observation: Observation) -> chex.Array:
        encoded_observation = self._encode_observation(observation)
        hidden_state = self._to_hidden(encoded_observation)
        if self.nonlinear_to_hidden:
            hidden_state = parse_activation_fn(self.recurrent_activation)(hidden_state)
        return hidden_state

    def _maybe_normalize_hidden_state(self, hidden_state: chex.Array) -> chex.Array:
        if self.normalize_hidden_state:
            max_hidden_state = jnp.max(hidden_state, axis=-1, keepdims=True)
            min_hidden_state = jnp.min(hidden_state, axis=-1, keepdims=True)
            hidden_state_range = max_hidden_state - min_hidden_state
            hidden_state = (hidden_state - min_hidden_state) / hidden_state_range * 2.0 - 1.0
        return hidden_state

    def recurrent_inference(self, hidden_state: chex.Array, action: chex.Array) -> chex.Array:
        if self.embed_actions:
            if action.dtype == jnp.int32:
                action = jax.nn.one_hot(action, self.action_dim)
            embedded_action = self._action_embeddings(action)
        else:
            if action.dtype == jnp.int32:
                action = jax.nn.one_hot(action, self.action_dim)
            embedded_action = action

        # Normalize hidden state
        hidden_state = self._maybe_normalize_hidden_state(hidden_state)

        # Run the RNN
        rnn_state = self._flat_to_rnn(hidden_state)
        next_rnn_state, rnn_output = self._core(rnn_state, embedded_action)
        next_hidden_state = self._rnn_to_flat(next_rnn_state)

        # Add residual connection
        next_hidden_state = next_hidden_state + hidden_state

        # Compute reward
        reward = self.reward_head(self.reward_torso(rnn_output))

        return next_hidden_state, reward

    def __call__(self, observation: Observation, action: chex.Array) -> chex.Array:
        """Mainly used for initialisation."""
        hidden_state = self.initial_inference(observation)
        next_hidden_state, reward = self.recurrent_inference(hidden_state, action)
        return next_hidden_state, reward
