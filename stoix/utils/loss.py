from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import rlax
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax.distributions import Distribution

tfd = tfp.distributions

# These losses are generally taken from rlax but edited to explicitly take in a batch of data.
# This is because the original rlax losses are not batched and are meant to be used with vmap,
# which is much slower.


def ppo_clip_loss(
    pi_log_prob_t: chex.Array, b_pi_log_prob_t: chex.Array, gae_t: chex.Array, epsilon: float
) -> chex.Array:
    ratio = jnp.exp(pi_log_prob_t - b_pi_log_prob_t)
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


def ppo_penalty_loss(
    pi_log_prob_t: chex.Array,
    b_pi_log_prob_t: chex.Array,
    gae_t: chex.Array,
    beta: float,
    pi: Distribution,
    b_pi: Distribution,
) -> Tuple[chex.Array, chex.Array]:
    ratio = jnp.exp(pi_log_prob_t - b_pi_log_prob_t)
    kl_div = b_pi.kl_divergence(pi).mean()
    objective = ratio * gae_t - beta * kl_div
    loss_actor = -objective.mean()
    return loss_actor, kl_div


def dpo_loss(
    pi_log_prob_t: chex.Array,
    b_pi_log_prob_t: chex.Array,
    gae_t: chex.Array,
    alpha: float,
    beta: float,
) -> chex.Array:
    log_diff = pi_log_prob_t - b_pi_log_prob_t
    ratio = jnp.exp(log_diff)
    is_pos = (gae_t >= 0.0).astype(jnp.float32)
    r1 = ratio - 1.0
    drift1 = jax.nn.relu(r1 * gae_t - alpha * jax.nn.tanh(r1 * gae_t / alpha))
    drift2 = jax.nn.relu(log_diff * gae_t - beta * jax.nn.tanh(log_diff * gae_t / beta))
    drift = drift1 * is_pos + drift2 * (1 - is_pos)
    loss_actor = -(ratio * gae_t - drift).mean()
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

    return td_error


def q_learning(
    q_tm1: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    q_t: chex.Array,
    huber_loss_parameter: chex.Array,
) -> jnp.ndarray:
    """Computes the double Q-learning loss. Each input is a batch."""
    batch_indices = jnp.arange(a_tm1.shape[0])
    # Compute Q-learning n-step TD-error.
    target_tm1 = r_t + d_t * jnp.max(q_t, axis=-1)
    td_error = target_tm1 - q_tm1[batch_indices, a_tm1]
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)

    return jnp.mean(batch_loss)


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
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)

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
    td_errors = target_tm1 - v_tm1
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_errors, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_errors)
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
    td_error = target_q - q_tm1_a
    if huber_loss_parameter > 0.0:
        batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)
    else:
        batch_loss = rlax.l2_loss(td_error)
    batch_loss = jnp.mean(batch_loss)
    return batch_loss


def quantile_regression_loss(
    dist_src: chex.Array,
    tau_src: chex.Array,
    dist_target: chex.Array,
    huber_param: float = 0.0,
) -> chex.Array:
    """Compute (Huber) QR loss between two discrete quantile-valued distributions.

    See "Distributional Reinforcement Learning with Quantile Regression" by
    Dabney et al. (https://arxiv.org/abs/1710.10044).

    Args:
        dist_src: source probability distribution.
        tau_src: source distribution probability thresholds.
        dist_target: target probability distribution.
        huber_param: Huber loss parameter, defaults to 0 (no Huber loss).
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.

    Returns:
        Quantile regression loss.
    """

    batch_indices = jnp.arange(dist_src.shape[0])

    # Calculate quantile error.
    delta = dist_target[batch_indices, None, :] - dist_src[batch_indices, :, None]
    delta_neg = (delta < 0.0).astype(jnp.float32)
    delta_neg = jax.lax.stop_gradient(delta_neg)
    weight = jnp.abs(tau_src[batch_indices, :, None] - delta_neg)

    # Calculate Huber loss.
    if huber_param > 0.0:
        loss = rlax.huber_loss(delta, huber_param)
    else:
        loss = jnp.abs(delta)
    loss *= weight

    # Average over target-samples dimension, sum over src-samples dimension.
    return jnp.sum(jnp.mean(loss, axis=-1), axis=-1)


def quantile_q_learning(
    dist_q_tm1: chex.Array,
    tau_q_tm1: chex.Array,
    a_tm1: chex.Array,
    r_t: chex.Array,
    d_t: chex.Array,
    dist_q_t_selector: chex.Array,
    dist_q_t: chex.Array,
    huber_param: float = 0.0,
) -> chex.Array:
    """Implements Q-learning for quantile-valued Q distributions.

    See "Distributional Reinforcement Learning with Quantile Regression" by
    Dabney et al. (https://arxiv.org/abs/1710.10044).

    Args:
        dist_q_tm1: Q distribution at time t-1.
        tau_q_tm1: Q distribution probability thresholds.
        a_tm1: action index at time t-1.
        r_t: reward at time t.
        d_t: discount at time t.
        dist_q_t_selector: Q distribution at time t for selecting greedy action in
        target policy. This is separate from dist_q_t as in Double Q-Learning, but
        can be computed with the target network and a separate set of samples.
        dist_q_t: target Q distribution at time t.
        huber_param: Huber loss parameter, defaults to 0 (no Huber loss).
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.

    Returns:
        Quantile regression Q learning loss.
    """
    batch_indices = jnp.arange(a_tm1.shape[0])

    # Only update the taken actions.
    dist_qa_tm1 = dist_q_tm1[batch_indices, :, a_tm1]

    # Select target action according to greedy policy w.r.t. dist_q_t_selector.
    q_t_selector = jnp.mean(dist_q_t_selector, axis=1)
    a_t = jnp.argmax(q_t_selector, axis=-1)
    dist_qa_t = dist_q_t[batch_indices, :, a_t]

    # Compute target, do not backpropagate into it.
    dist_target = r_t[:, jnp.newaxis] + d_t[:, jnp.newaxis] * dist_qa_t
    dist_target = jax.lax.stop_gradient(dist_target)

    return quantile_regression_loss(dist_qa_tm1, tau_q_tm1, dist_target, huber_param).mean()
