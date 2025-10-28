from typing import Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import Initializer, orthogonal

from stoix.networks.utils import parse_activation_fn
from stoix.systems.disco_rl.disco_rl_types import AgentOutput


class LSTMActionConditionedTorso(nn.Module):
    """LSTM-based action-conditional torso inspired by Muesli/MuZero.

    This torso creates a root embedding from the observation, then performs
    an LSTM transition for all possible actions in parallel, producing
    action-conditional hidden states of shape [batch, num_actions, hidden_dim].

    Attributes:
        num_actions: Number of discrete actions.
        lstm_size: Size of the LSTM hidden state.
        root_mlp_sizes: Sizes of MLP layers for root embedding. If None, uses a single linear layer.
        activation: Activation function for the root MLP.
        kernel_init: Kernel initializer for linear layers.
    """

    num_actions: int
    lstm_size: int
    root_mlp_sizes: Tuple[int, ...] = ()
    activation: str = "relu"
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass.

        Args:
            observation: Input observation of shape [batch, ...].

        Returns:
            Action-conditional hidden states of shape [batch, num_actions, lstm_size].
        """
        batch_size = observation.shape[0]

        # 1. Create root embedding from observation
        root_embedding = self._root_embedding(observation)  # [batch, lstm_size]

        # 2. Perform LSTM transition for all actions
        action_hidden_states = self._model_transition_all_actions(
            root_embedding, batch_size
        )  # [batch, num_actions, lstm_size]

        return action_hidden_states

    def _root_embedding(self, observation: chex.Array) -> chex.Array:
        """Constructs a root embedding from the observation.

        Args:
            observation: Input observation of shape [batch, ...].

        Returns:
            Root embedding (LSTM cell state) of shape [batch, lstm_size].
        """
        # Simply use the observation as input
        x = observation

        # Apply optional MLP layers
        if self.root_mlp_sizes:
            for size in self.root_mlp_sizes:
                x = nn.Dense(size, kernel_init=self.kernel_init)(x)
                x = parse_activation_fn(self.activation)(x)

        # Final linear layer to get cell state
        cell = nn.Dense(self.lstm_size, kernel_init=self.kernel_init, name="root_cell")(x)
        # Create hidden state as tanh(cell)
        hidden = jnp.tanh(cell)
        return (hidden, cell)

    def _model_transition_all_actions(self, root_carry: chex.Array, batch_size: int) -> chex.Array:
        """Performs LSTM transition for all actions in parallel.

        Args:
            root_carry: Root carry state of shape [batch, lstm_size].
            batch_size: Batch size.

        Returns:
            LSTM outputs for all actions of shape [batch, num_actions, lstm_size].
        """
        # Create one-hot encodings for all actions
        # Shape: [num_actions, num_actions]
        one_hot_actions = jnp.eye(self.num_actions, dtype=root_carry[0].dtype)

        # Repeat for each batch element
        # Shape: [batch * num_actions, num_actions]
        batched_one_hot_actions = jnp.tile(one_hot_actions, [batch_size, 1])

        # Repeat the root carry for each action
        # This uses jax.tree.map to handle the (hidden, cell) tuple
        initial_carry = jax.tree.map(
            lambda x: jnp.repeat(x, repeats=self.num_actions, axis=0), root_carry
        )

        # Apply LSTM
        lstm_cell = nn.LSTMCell(features=self.lstm_size, name="action_cond_lstm")
        _, lstm_output = lstm_cell(initial_carry, batched_one_hot_actions)

        # Reshape output from [batch * num_actions, lstm_size] to [batch, num_actions, lstm_size]
        action_hidden_states = lstm_output.reshape(batch_size, self.num_actions, self.lstm_size)

        return action_hidden_states


class DiscoAgentNetwork(nn.Module):
    """
    A network for the DiscoRL agent.

    This network has a shared torso and five separate heads, matching
    the architecture required by the DiscoUpdateRule:
    1. logits (Policy)
    2. q (Categorical Value)
    3. y (Auxiliary)
    4. z (Auxiliary)
    5. aux_pi (Auxiliary Policy)
    """

    shared_torso: nn.Module
    action_conditional_torso: nn.Module
    logits_head: nn.Module
    q_head: nn.Module
    y_head: nn.Module
    z_head: nn.Module
    aux_pi_head: nn.Module

    def __call__(self, obs: chex.Array) -> AgentOutput:
        """Forward pass."""
        # Run the shared torso
        torso_output = self.shared_torso(obs)

        # Run logits and y prediction heads on the torso output
        logits = self.logits_head(torso_output)
        y = self.y_head(torso_output)

        # We now run the action conditional heads.
        # We do this by running an action-conditional torso first,
        # then passing its output to the q, z, and aux_pi heads.
        action_conditional_torso_output = self.action_conditional_torso(torso_output)
        q = self.q_head(action_conditional_torso_output)
        z = self.z_head(action_conditional_torso_output)
        aux_pi = self.aux_pi_head(action_conditional_torso_output)

        return AgentOutput(logits=logits, q=q, y=y, z=z, aux_pi=aux_pi)
