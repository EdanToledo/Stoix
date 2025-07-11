import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from kinetix.models.actor_critic import MultiHeadDense
from kinetix.render.renderer_symbolic_entity import EntityObservation

from stoix.networks.utils import parse_activation_fn


class PermutationInvariantEntityEncoder(nn.Module):
    """An encoder for Kinetix's symbolic entity observations,
        that is invariant to the ordering of entities.

    Arguments:
        activation: The activation function to use in the encoder.
        num_heads: The number of heads to use in the multi-head dense layer.
        hidden_dim: The final hidden dimension of the encoder.
        entity_encoder_dim: How many features to encode each entity with
    """

    activation: str = "tanh"
    num_heads: int = 4
    hidden_dim: int = 256
    entity_encoder_dim: int = 64

    @nn.compact
    def __call__(self, obs: EntityObservation | dict[str, chex.Array]) -> chex.Array:
        activation_fn = parse_activation_fn(self.activation)
        if isinstance(obs, dict):
            obs = EntityObservation(**obs)

        assert obs.circles.ndim == 3, f"{obs.circles.shape=} must be 3D, got {obs.circles.ndim=}"
        assert (
            self.hidden_dim % self.num_heads == 0
        ), f"{self.hidden_dim=} must be divisible by {self.num_heads=}"

        def _single_encoder(features: chex.Array, entity_id: int) -> chex.Array:
            num_to_remove = 4
            embedding = activation_fn(
                nn.Dense(
                    self.entity_encoder_dim - num_to_remove,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(features)
            )
            id_1h = jnp.zeros((*embedding.shape[:2], 4)).at[..., entity_id].set(1)
            return jnp.concatenate([embedding, id_1h], axis=-1)

        circle_encodings = _single_encoder(obs.circles, 0)
        polygon_encodings = _single_encoder(obs.polygons, 1)
        joint_encodings = _single_encoder(obs.joints, 2)
        thruster_encodings = _single_encoder(obs.thrusters, 3)

        all_encodings = jnp.concatenate(
            [polygon_encodings, circle_encodings, joint_encodings, thruster_encodings], axis=1
        )
        all_mask = jnp.concatenate(
            [obs.polygon_mask, obs.circle_mask, obs.joint_mask, obs.thruster_mask], axis=1
        )

        def mask(features: chex.Array, mask: chex.Array) -> chex.Array:
            return jnp.where(mask[:, None], features, jnp.zeros_like(features))

        obs = jax.vmap(mask)(all_encodings, all_mask)

        dim_per_head = self.hidden_dim // self.num_heads
        embedding = MultiHeadDense(
            num_heads=self.num_heads,
            out_dim=dim_per_head,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)

        return embedding
