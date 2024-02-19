import chex
import jax
import jax.numpy as jnp
import rlax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


def ppo_loss(
    pi_log_prob_t: chex.Array, b_pi_log_prob_t: chex.Array, gae_t: chex.Array, epsilon: float
) -> chex.Array:
    ratio = jnp.exp(pi_log_prob_t - b_pi_log_prob_t)
    gae_t = (gae_t - gae_t.mean()) / (gae_t.std() + 1e-8)
    loss_actor1 = ratio * gae_t
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - epsilon,
            1.0 + epsilon,
        )
        * gae_t
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    return loss_actor


def clipped_value_loss(
    pred_value_t: chex.Array, behavior_value_t: chex.Array, targets_t: chex.Array, epsilon: float
) -> chex.Array:
    value_pred_clipped = behavior_value_t + (pred_value_t - behavior_value_t).clip(
        -epsilon, epsilon
    )
    value_losses = jnp.square(pred_value_t - targets_t)
    value_losses_clipped = jnp.square(value_pred_clipped - targets_t)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    return value_loss


def categorical_double_q_learning(
    q_logits_tm1: chex.Array,
    q_atoms_tm1: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_logits_t: chex.Array,
    q_atoms_t: chex.Array,
    q_t_selector: chex.Array,
) -> chex.Array:
    """Computes the categorical double Q-learning loss. Each input is a batch."""
    batch_indices = jnp.arange(a_tm1.shape[0])
    # Scale and shift time-t distribution atoms by discount and reward.
    target_z = r_t[:, jnp.newaxis] + d_t[:, jnp.newaxis] * q_atoms_t
    # Select logits for greedy action in state s_t and convert to distribution.
    p_target_z = jax.nn.softmax(q_logits_t[batch_indices, q_t_selector.argmax(-1)])
    # Project using the Cramer distance and maybe stop gradient flow to targets.
    target = jax.vmap(rlax.categorical_l2_project)(target_z, p_target_z, q_atoms_tm1)
    # Compute loss (i.e. temporal difference error).
    logit_qa_tm1 = q_logits_tm1[batch_indices, a_tm1]
    td_error = tfd.Categorical(probs=target).cross_entropy(tfd.Categorical(logits=logit_qa_tm1))
    q_loss = jnp.mean(td_error)

    return q_loss


def double_q_learning(
    q_tm1: chex.Array,
    q_t_value: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_t_selector: chex.Array,
    huber_loss_parameter: chex.Array,
) -> jnp.ndarray:
    """Computes the double Q-learning loss. Each input is a batch."""
    batch_indices = jnp.arange(a_tm1.shape[0])
    # Compute double Q-learning n-step TD-error.
    target_tm1 = r_t + d_t * q_t_value[batch_indices, q_t_selector.argmax(-1)]
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]
    batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)

    return jnp.mean(batch_loss)


def categorical_td_learning(
    v_logits_tm1: chex.Array,
    v_atoms_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    v_logits_t: chex.Array,
    v_atoms_t: chex.Array,
) -> chex.Array:
    """Implements TD-learning for categorical value distributions. Each input is a batch."""

    # Scale and shift time-t distribution atoms by discount and reward.
    target_z = r_t[:, jnp.newaxis] + d_t[:, jnp.newaxis] * v_atoms_t

    # Convert logits to distribution.
    v_t_probs = jax.nn.softmax(v_logits_t)

    # Project using the Cramer distance and maybe stop gradient flow to targets.
    target = jax.vmap(rlax.categorical_l2_project)(target_z, v_t_probs, v_atoms_tm1)

    td_error = tfd.Categorical(probs=target).cross_entropy(tfd.Categorical(logits=v_logits_tm1))

    return jnp.mean(td_error)
