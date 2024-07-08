from typing import Sequence, Tuple, Union

import chex
import distrax
import jax.numpy as jnp
from flax import linen as nn

from stoix.base_types import Observation, RNNObservation
from stoix.networks.inputs import ObservationInput


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

        obs_embedding = self.input_layer(observation)
        obs_embedding = self.torso(obs_embedding)
        critic_output = self.critic_head(obs_embedding)

        return critic_output


class CompositeNetwork(nn.Module):
    """Composite Network. Takes in a sequence of layers and applies them sequentially."""

    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(
        self, *network_input: Union[chex.Array, Tuple[chex.Array, ...]]
    ) -> Union[distrax.DistributionLike, chex.Array]:

        x = self.layers[0](*network_input)
        for layer in self.layers[1:]:
            x = layer(x)
        return x


class MultiNetwork(nn.Module):
    """Multi Network.

    Takes in a sequence of networks, applies them separately and concatenates the outputs."""

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


class RecurrentActor(nn.Module):
    """Recurrent Actor Network."""

    action_head: nn.Module
    post_torso: nn.Module
    rnn: nn.Module
    pre_torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        policy_hidden_state: chex.Array,
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, distrax.DistributionLike]:

        observation, done = observation_done

        observation = self.input_layer(observation)
        policy_embedding = self.pre_torso(observation)
        policy_rnn_input = (policy_embedding, done)
        policy_hidden_state, policy_embedding = self.rnn(policy_hidden_state, policy_rnn_input)
        actor_logits = self.post_torso(policy_embedding)
        pi = self.action_head(actor_logits)

        return policy_hidden_state, pi


class RecurrentCritic(nn.Module):
    """Recurrent Critic Network."""

    critic_head: nn.Module
    post_torso: nn.Module
    rnn: nn.Module
    pre_torso: nn.Module
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(
        self,
        critic_hidden_state: Tuple[chex.Array, chex.Array],
        observation_done: RNNObservation,
    ) -> Tuple[chex.Array, chex.Array]:

        observation, done = observation_done

        observation = self.input_layer(observation)

        critic_embedding = self.pre_torso(observation)
        critic_rnn_input = (critic_embedding, done)
        critic_hidden_state, critic_embedding = self.rnn(critic_hidden_state, critic_rnn_input)
        critic_output = self.post_torso(critic_embedding)
        critic_output = self.critic_head(critic_output)

        return critic_hidden_state, critic_output
