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


def td_learning(
    v_tm1: chex.Array,
    r_t: chex.Array,
    discount_t: chex.Array,
    v_t: chex.Array,
    huber_loss_parameter: chex.Array,
) -> chex.Array:
    """Calculates the temporal difference error. Each input is a batch."""
    target_tm1 = r_t + discount_t * v_t
    batch_loss = rlax.huber_loss(target_tm1 - v_tm1, huber_loss_parameter)
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


def munchausen_q_learning(
    q_tm1: chex.Array,
    q_tm1_target: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_t_target: chex.Array,
    entropy_temperature: chex.Array,
    munchausen_coefficient: chex.Array,
    clip_value_min: chex.Array,
    huber_loss_parameter: chex.Array,
) -> chex.Array:
    action_one_hot = jax.nn.one_hot(a_tm1, q_tm1.shape[-1])
    q_tm1_a = jnp.sum(q_tm1 * action_one_hot, axis=-1)
    # Compute double Q-learning loss.
    # Munchausen term : tau * log_pi(a|s)
    munchausen_term = entropy_temperature * jax.nn.log_softmax(
        q_tm1_target / entropy_temperature, axis=-1
    )
    munchausen_term_a = jnp.sum(action_one_hot * munchausen_term, axis=-1)
    munchausen_term_a = jnp.clip(munchausen_term_a, a_min=clip_value_min, a_max=0.0)

    # Soft Bellman operator applied to q
    next_v = entropy_temperature * jax.nn.logsumexp(q_t_target / entropy_temperature, axis=-1)
    target_q = jax.lax.stop_gradient(
        r_t + munchausen_coefficient * munchausen_term_a + d_t * next_v
    )

    batch_loss = rlax.huber_loss(target_q - q_tm1_a, huber_loss_parameter)
    batch_loss = jnp.mean(batch_loss)
    return batch_loss
