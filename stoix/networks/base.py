import functools
from typing import Sequence, Tuple, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from stoix.networks.inputs import ObservationInput
from stoix.types import Observation, RNNObservation


class FeedForwardActor(nn.Module):
    """Simple Feedforward Actor Network."""

    action_head: nn.Module
    torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(self, observation: Observation) -> distrax.DistributionLike:

        obs_embedding = self.input_layer(observation)

        obs_embedding = self.torso(obs_embedding)

        return self.action_head(obs_embedding)


class FeedForwardCritic(nn.Module):
    """Simple Feedforward Critic Network."""

    critic_head: nn.Module
    torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        """Forward pass."""

        obs_embedding = self.input_layer(observation)
        obs_embedding = self.torso(obs_embedding)
        critic_output = self.critic_head(obs_embedding)

        return critic_output


class CompositeNetwork(nn.Module):
    """Composite Network."""

    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(
        self, *network_input: Union[chex.Array, Tuple[chex.Array, ...]]
    ) -> Union[distrax.DistributionLike, chex.Array]:
        """Forward pass."""
        x = self.layers[0](*network_input)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class MultiNetwork(nn.Module):
    """Multi Network."""

    networks: Sequence[nn.Module]

    @nn.compact
    def __call__(
        self, *network_input: Union[chex.Array, Tuple[chex.Array, ...]]
    ) -> Union[distrax.DistributionLike, chex.Array]:
        """Forward pass."""
        outputs = []
        for network in self.networks:
            outputs.append(network(*network_input))
        concatenated = jnp.stack(outputs, axis=-1)
        chex.assert_rank(concatenated, 2)
        return concatenated


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class RecurrentActor(nn.Module):
    """Recurrent Actor Network."""

    action_head: nn.Module
    post_torso: nn.Module
    pre_torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, distrax.Categorical]:
        """Forward pass."""
        observation, done = observation_done

        observation = self.input_layer(observation)
        policy_embedding = self.pre_torso(observation)
        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = ScannedRNN()(policy_hidden_state, policy_rnn_input)
        actor_logits = self.post_torso(policy_embedding)
        pi = self.action_head(actor_logits)

        return policy_hidden_state, pi


class RecurrentCritic(nn.Module):
    """Recurrent Critic Network."""

    critic_head: nn.Module
    post_torso: nn.Module
    pre_torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass."""
        observation, done = observation_done

        observation = self.input_layer(observation)

        critic_embedding = self.pre_torso(observation)
        critic_rnn_input = (critic_embedding, done)
        critic_hidden_state, critic_embedding = ScannedRNN()(critic_hidden_state, critic_rnn_input)
        critic_output = self.post_torso(critic_embedding)
        critic_output = self.critic_head(critic_output)

        return critic_hidden_state, critic_output
