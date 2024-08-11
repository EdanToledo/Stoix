import functools
from typing import Sequence, Tuple, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from stoix.base_types import Observation, RNNObservation
from stoix.networks.inputs import ObservationInput
from stoix.networks.utils import parse_rnn_cell


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


class ScannedRNN(nn.Module):
    hidden_state_dim: int
    cell_type: str

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, rnn_state: chex.Array, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        ins, resets = x
        hidden_state_reset_fn = lambda reset_state, current_state: jnp.where(
            resets[:, np.newaxis],
            reset_state,
            current_state,
        )
        rnn_state = jax.tree_util.tree_map(
            hidden_state_reset_fn,
            self.initialize_carry(ins.shape[0]),
            rnn_state,
        )
        new_rnn_state, y = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)(
            rnn_state, ins
        )
        return new_rnn_state, y

    @nn.nowrap
    def initialize_carry(self, batch_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        cell = parse_rnn_cell(self.cell_type)(features=self.hidden_state_dim)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, self.hidden_state_dim))


class RecurrentActor(nn.Module):
    """Recurrent Actor Network."""

    action_head: nn.Module
    post_torso: nn.Module
    hidden_state_dim: int
    cell_type: str
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
        policy_hidden_state, policy_embedding = ScannedRNN(self.hidden_state_dim, self.cell_type)(
            policy_hidden_state, policy_rnn_input
        )
        actor_logits = self.post_torso(policy_embedding)
        pi = self.action_head(actor_logits)

        return policy_hidden_state, pi


class RecurrentCritic(nn.Module):
    """Recurrent Critic Network."""

    critic_head: nn.Module
    post_torso: nn.Module
    hidden_state_dim: int
    cell_type: str
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
        critic_hidden_state, critic_embedding = ScannedRNN(self.hidden_state_dim, self.cell_type)(
            critic_hidden_state, critic_rnn_input
        )
        critic_output = self.post_torso(critic_embedding)
        critic_output = self.critic_head(critic_output)

        return critic_hidden_state, critic_output
