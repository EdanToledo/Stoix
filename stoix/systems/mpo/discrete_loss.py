from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import Categorical

from stoix.systems.mpo.mpo_types import CategoricalDualParams

# These functions are largely taken from Acme's MPO implementation:

_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0

Shape = Tuple[int]
DType = type(jnp.float32)


def categorical_mpo_loss(
    dual_params: CategoricalDualParams,
    online_action_distribution: Categorical,
    target_action_distribution: Categorical,
    q_values: chex.Array,  # Shape [D, B].
    epsilon: float,
    epsilon_policy: float,
) -> Tuple[chex.Array, chex.ArrayTree]:
    """Computes the MPO loss for a categorical policy.

    Args:
        dual_params: parameters tracking the temperature and the dual variables.
        online_action_distribution: online distribution returned by the online
            policy network; expects batch_dims of [B] and event_dims of [D].
        target_action_distribution: target distribution returned by the target
            policy network; expects same shapes as online distribution.
        q_values: Q-values associated with every action; expects shape [D, B].
        epsilon: KL constraint on the non-parametric auxiliary policy, the one
            associated with the dual variable called temperature.
        epsilon_policy: KL constraint on the categorical policy, the one
            associated with the dual variable called alpha.


    Returns:
      Loss, combining the policy loss, KL penalty, and dual losses required to
        adapt the dual variables.
      Stats, for diagnostics and tracking performance.
    """

    q_values = jnp.transpose(q_values)  # [D, B] --> [B, D].

    # Transform dual variables from log-space.
    # Note: using softplus instead of exponential for numerical stability.
    temperature = get_temperature_from_params(dual_params).squeeze()
    alpha = jax.nn.softplus(dual_params.log_alpha).squeeze() + _MPO_FLOAT_EPSILON

    # Compute the E-step logits and the temperature loss, used to adapt the
    # tempering of Q-values.
    (
        logits_e_step,
        loss_temperature,
    ) = compute_weights_and_temperature_loss(  # pytype: disable=wrong-arg-types  # jax-ndarray
        q_values=q_values,
        logits=target_action_distribution.logits,
        epsilon=epsilon,
        temperature=temperature,
    )
    action_distribution_e_step = Categorical(logits=logits_e_step)

    # Only needed for diagnostics: Compute estimated actualized KL between the
    # non-parametric and current target policies.
    kl_nonparametric = action_distribution_e_step.kl_divergence(target_action_distribution)

    # Compute the policy loss.
    loss_policy = action_distribution_e_step.cross_entropy(online_action_distribution)
    loss_policy = jnp.mean(loss_policy)

    # Compute the regularization.
    kl = target_action_distribution.kl_divergence(online_action_distribution)
    mean_kl = jnp.mean(kl, axis=0)
    loss_kl = jax.lax.stop_gradient(alpha) * mean_kl

    # Compute the dual loss.
    loss_alpha = alpha * (epsilon_policy - jax.lax.stop_gradient(mean_kl))

    # Combine losses.
    loss_dual = loss_alpha + loss_temperature
    loss = loss_policy + loss_kl + loss_dual

    # Create statistics.
    loss_info = {
        "temperature": temperature,
        "alpha": alpha,
        "loss_temperature": loss_temperature.mean(),
        "loss_alpha": loss_alpha.mean(),
        "loss_policy": loss_policy.mean(),
        "loss_kl": loss_kl.mean(),
        "kl_nonparametric": kl_nonparametric.mean(),
        "entropy_online": online_action_distribution.entropy().mean(),
        "entropy_target": target_action_distribution.entropy().mean(),
        "kl_mean_rel": mean_kl / epsilon_policy,
        "kl_q_rel": jnp.mean(kl_nonparametric) / epsilon,
        "q_min": jnp.mean(jnp.min(q_values, axis=0)),
        "q_max": jnp.mean(jnp.max(q_values, axis=0)),
    }

    return loss, loss_info


def compute_weights_and_temperature_loss(
    q_values: chex.Array,
    logits: chex.Array,
    epsilon: float,
    temperature: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Computes normalized importance weights for the policy optimization.

    Args:
      q_values: Q-values associated with the actions sampled from the target
        policy; expected shape [B, D].
      logits: Parameters to the categorical distribution with respect to which the
        expectations are going to be computed.
      epsilon: Desired constraint on the KL between the target and non-parametric
        policies.
      temperature: Scalar used to temper the Q-values before computing normalized
        importance weights from them. This is really the Lagrange dual variable in
        the constrained optimization problem, the solution of which is the
        non-parametric policy targeted by the policy loss.

    Returns:
      Normalized importance weights, used for policy optimization.
      Temperature loss, used to adapt the temperature.
    """

    # Temper the given Q-values using the current temperature.
    tempered_q_values = jax.lax.stop_gradient(q_values) / temperature

    # Compute the E-step normalized logits.
    unnormalized_logits = tempered_q_values + jax.nn.log_softmax(logits, axis=-1)
    logits_e_step = jax.nn.log_softmax(unnormalized_logits, axis=-1)

    # Compute the temperature loss (dual of the E-step optimization problem).
    # Note that the log normalizer will be the same for all actions, so we choose
    # only the first one.
    log_normalizer = unnormalized_logits[:, 0] - logits_e_step[:, 0]
    loss_temperature = temperature * (epsilon + jnp.mean(log_normalizer))

    return logits_e_step, loss_temperature


def clip_categorical_mpo_params(params: CategoricalDualParams) -> CategoricalDualParams:
    return params._replace(
        log_temperature=jnp.maximum(_MIN_LOG_TEMPERATURE, params.log_temperature),
        log_alpha=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha),
    )


def get_temperature_from_params(params: CategoricalDualParams) -> chex.Array:
    return jax.nn.softplus(params.log_temperature) + _MPO_FLOAT_EPSILON
