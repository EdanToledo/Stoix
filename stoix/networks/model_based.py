from typing import Callable

import chex
from flax import linen as nn

from stoix.base_types import Observation
from stoix.networks.inputs import ObservationInput
from stoix.networks.postprocessors import min_max_normalize


class Representation(nn.Module):
    torso: nn.Module
    embedding_head: nn.Module
    post_processor: Callable[[chex.Array], chex.Array] = min_max_normalize
    input_layer: nn.Module = ObservationInput()

    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        observation = self.input_layer(observation)
        representation = self.torso(observation)
        representation = self.embedding_head(representation)
        return self.post_processor(representation)


class Dynamics(nn.Module):
    torso: nn.Module
    embedding_head: nn.Module
    reward_head: nn.Module
    input_layer: nn.Module
    embedding_post_processor: Callable[[chex.Array], chex.Array] = min_max_normalize

    @nn.compact
    def __call__(self, embedding: chex.Array, action: chex.Array) -> chex.Array:
        embedding = self.input_layer(embedding, action)
        dynamics_embedding = self.torso(embedding)
        next_embedding = self.embedding_head(dynamics_embedding)
        reward = self.reward_head(dynamics_embedding)
        return self.embedding_post_processor(next_embedding), reward
