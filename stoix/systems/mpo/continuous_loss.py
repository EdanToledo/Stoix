from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import (
    Distribution,
    Independent,
    MultivariateNormalDiag,
    Normal,
)

from stoix.systems.mpo.types import DualParams, MPOStats

# These functions are largely taken from Acme's MPO implementation:

_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0

Shape = Tuple[int]
DType = type(jnp.float32)


def compute_weights_and_temperature_loss(
    q_values: chex.Array,
    epsilon: float,
    temperature: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Computes normalized importance weights for the policy optimization.

    Args:
      q_values: Q-values associated with the actions sampled from the target
        policy; expected shape [N, B].
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

    # Compute the normalized importance weights used to compute expectations with
    # respect to the non-parametric policy.
    normalized_weights = jax.nn.softmax(tempered_q_values, axis=0)
    normalized_weights = jax.lax.stop_gradient(normalized_weights)

    # Compute the temperature loss (dual of the E-step optimization problem).
    q_logsumexp = jax.scipy.special.logsumexp(tempered_q_values, axis=0)
    log_num_actions = jnp.log(q_values.shape[0] / 1.0)
    loss_temperature = epsilon + jnp.mean(q_logsumexp) - log_num_actions
    loss_temperature = temperature * loss_temperature

    return normalized_weights, loss_temperature


def compute_nonparametric_kl_from_normalized_weights(
    normalized_weights: chex.Array,
) -> chex.Array:
    """Estimate the actualized KL between the non-parametric and target policies."""

    # Compute integrand.
    num_action_samples = normalized_weights.shape[0] / 1.0
    integrand = jnp.log(num_action_samples * normalized_weights + 1e-8)

    # Return the expectation with respect to the non-parametric policy.
    return jnp.sum(normalized_weights * integrand, axis=0)


def compute_cross_entropy_loss(
    sampled_actions: chex.Array,
    normalized_weights: chex.Array,
    online_action_distribution: Distribution,
) -> chex.Array:
    """Compute cross-entropy online and the reweighted target policy.

    Args:
      sampled_actions: samples used in the Monte Carlo integration in the policy
        loss. Expected shape is [N, B, ...], where N is the number of sampled
        actions and B is the number of sampled states.
      normalized_weights: target policy multiplied by the exponentiated Q values
        and normalized; expected shape is [N, B].
      online_action_distribution: policy to be optimized.

    Returns:
      loss_policy_gradient: the cross-entropy loss that, when differentiated,
        produces the policy gradient.
    """

    # Compute the M-step loss.
    log_prob = online_action_distribution.log_prob(sampled_actions)

    # Compute the weighted average log-prob using the normalized weights.
    loss_policy_gradient = -jnp.sum(log_prob * normalized_weights, axis=0)

    # Return the mean loss over the batch of states.
    return jnp.mean(loss_policy_gradient, axis=0)


def compute_parametric_kl_penalty_and_dual_loss(
    kl: chex.Array,
    alpha: chex.Array,
    epsilon: float,
) -> Tuple[chex.Array, chex.Array]:
    """Computes the KL cost to be added to the Lagragian and its dual loss.

    The KL cost is simply the alpha-weighted KL divergence and it is added as a
    regularizer to the policy loss. The dual variable alpha itself has a loss that
    can be minimized to adapt the strength of the regularizer to keep the KL
    between consecutive updates at the desired target value of epsilon.

    Args:
      kl: KL divergence between the target and online policies.
      alpha: Lagrange multipliers (dual variables) for the KL constraints.
      epsilon: Desired value for the KL.

    Returns:
      loss_kl: alpha-weighted KL regularization to be added to the policy loss.
      loss_alpha: The Lagrange dual loss minimized to adapt alpha.
    """

    # Compute the mean KL over the batch.
    mean_kl = jnp.mean(kl, axis=0)

    # Compute the regularization.
    loss_kl = jnp.sum(jax.lax.stop_gradient(alpha) * mean_kl)

    # Compute the dual loss.
    loss_alpha = jnp.sum(alpha * (epsilon - jax.lax.stop_gradient(mean_kl)))

    return loss_kl, loss_alpha


def clip_dual_params(params: DualParams, per_dim_constraining: bool) -> DualParams:
    clipped_params = DualParams(
        log_temperature=jnp.maximum(_MIN_LOG_TEMPERATURE, params.log_temperature),
        log_alpha_mean=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_mean),
        log_alpha_stddev=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_stddev),
    )
    if not per_dim_constraining:
        return clipped_params
    else:
        return clipped_params._replace(
            log_penalty_temperature=jnp.maximum(
                _MIN_LOG_TEMPERATURE, params.log_penalty_temperature
            )
        )


def mpo_loss(
    dual_params: DualParams,
    online_action_distribution: Union[MultivariateNormalDiag, Independent],
    target_action_distribution: Union[MultivariateNormalDiag, Independent],
    target_sampled_actions: chex.Array,  # Shape [N, B, D].
    target_sampled_q_values: chex.Array,  # Shape [N, B].
    epsilon: float,
    epsilon_mean: float,
    epsilon_stddev: float,
    per_dim_constraining: bool,
    action_penalization: bool,
    epsilon_penalty: float,
) -> Tuple[chex.Array, MPOStats]:
    """Computes the decoupled MPO loss.

    Args:
        dual_params: parameters tracking the temperature and the dual variables.
        online_action_distribution: online distribution returned by the online
        policy network; expects batch_dims of [B] and event_dims of [D].
        target_action_distribution: target distribution returned by the target
        policy network; expects same shapes as online distribution.
        target_sampled_actions: actions sampled from the target policy; expects shape [N, B, D].
        target_sampled_q_values: Q-values associated with each action; expects shape [N, B].
        epsilon: KL constraint on the non-parametric auxiliary policy, the one associated with the
            dual variable called temperature.
        epsilon_mean: KL constraint on the mean of the Gaussian policy, the one associated with the
            dual variable called alpha_mean.
        epsilon_stddev: KL constraint on the stddev of the Gaussian policy, the one associated with
            the dual variable called alpha_mean.
        per_dim_constraining: whether to enforce the KL constraint on each dimension independently;
            this is the default. Otherwise the overall KL is constrained, which allows some
            dimensions to change more at the expense of others staying put.
        action_penalization: whether to use a KL constraint to penalize actions via the MO-MPO
            algorithm.
        epsilon_penalty: KL constraint on the probability of violating the action constraint.

    Returns:
        Loss, combining the policy loss, KL penalty, and dual losses required to
        adapt the dual variables.
        Stats, for diagnostics and tracking performance.
    """

    # Cast `MultivariateNormalDiag`s to Independent Normals.
    # The latter allows us to satisfy KL constraints per-dimension.
    if isinstance(target_action_distribution, MultivariateNormalDiag):
        target_action_distribution = Independent(
            Normal(target_action_distribution.mean(), target_action_distribution.stddev())
        )
        online_action_distribution = Independent(
            Normal(online_action_distribution.mean(), online_action_distribution.stddev())
        )

    # Transform dual variables from log-space.
    # Note: using softplus instead of exponential for numerical stability.
    temperature = jax.nn.softplus(dual_params.log_temperature) + _MPO_FLOAT_EPSILON
    alpha_mean = jax.nn.softplus(dual_params.log_alpha_mean) + _MPO_FLOAT_EPSILON
    alpha_stddev = jax.nn.softplus(dual_params.log_alpha_stddev) + _MPO_FLOAT_EPSILON

    # Get online and target means and stddevs in preparation for decomposition.
    online_mean = online_action_distribution.distribution.mean()
    online_scale = online_action_distribution.distribution.stddev()
    target_mean = target_action_distribution.distribution.mean()
    target_scale = target_action_distribution.distribution.stddev()

    # Compute normalized importance weights, used to compute expectations with
    # respect to the non-parametric policy; and the temperature loss, used to
    # adapt the tempering of Q-values.
    normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
        target_sampled_q_values, epsilon, temperature
    )

    # Only needed for diagnostics: Compute estimated actualized KL between the
    # non-parametric and current target policies.
    kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(normalized_weights)

    if action_penalization:
        # Transform action penalization temperature.
        penalty_temperature = (
            jax.nn.softplus(dual_params.log_penalty_temperature) + _MPO_FLOAT_EPSILON
        )

        # Compute action penalization cost.
        # Note: the cost is zero in [-1, 1] and quadratic beyond.
        diff_out_of_bound = target_sampled_actions - jnp.clip(target_sampled_actions, -1.0, 1.0)
        cost_out_of_bound = -jnp.linalg.norm(diff_out_of_bound, axis=-1)

        penalty_normalized_weights, loss_penalty_temperature = compute_weights_and_temperature_loss(
            cost_out_of_bound, epsilon_penalty, penalty_temperature
        )

        # Only needed for diagnostics: Compute estimated actualized KL between the
        # non-parametric and current target policies.
        penalty_kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
            penalty_normalized_weights
        )

        # Combine normalized weights.
        normalized_weights += penalty_normalized_weights
        loss_temperature += loss_penalty_temperature

    # Decompose the online policy into fixed-mean & fixed-stddev distributions.
    # This has been documented as having better performance in bandit settings,
    # see e.g. https://arxiv.org/pdf/1812.02256.pdf.
    fixed_stddev_distribution = Independent(Normal(loc=online_mean, scale=target_scale))
    fixed_mean_distribution = Independent(Normal(loc=target_mean, scale=online_scale))

    # Compute the decomposed policy losses.
    loss_policy_mean = compute_cross_entropy_loss(
        target_sampled_actions, normalized_weights, fixed_stddev_distribution
    )
    loss_policy_stddev = compute_cross_entropy_loss(
        target_sampled_actions, normalized_weights, fixed_mean_distribution
    )

    # Compute the decomposed KL between the target and online policies.
    if per_dim_constraining:
        kl_mean = target_action_distribution.distribution.kl_divergence(
            fixed_stddev_distribution.distribution
        )  # Shape [B, D].
        kl_stddev = target_action_distribution.distribution.kl_divergence(
            fixed_mean_distribution.distribution
        )  # Shape [B, D].
    else:
        kl_mean = target_action_distribution.kl_divergence(fixed_stddev_distribution)  # Shape [B].
        kl_stddev = target_action_distribution.kl_divergence(fixed_mean_distribution)  # Shape [B].

    # Compute the alpha-weighted KL-penalty and dual losses to adapt the alphas.
    loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
        kl_mean, alpha_mean, epsilon_mean
    )
    loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
        kl_stddev, alpha_stddev, epsilon_stddev
    )

    # Combine losses.
    loss_policy = loss_policy_mean + loss_policy_stddev
    loss_kl_penalty = loss_kl_mean + loss_kl_stddev
    loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature
    loss = loss_policy + loss_kl_penalty + loss_dual

    # Create statistics.
    pi_stddev = online_action_distribution.distribution.stddev()
    stats = MPOStats(
        # Dual Variables.
        dual_alpha_mean=jnp.mean(alpha_mean),
        dual_alpha_stddev=jnp.mean(alpha_stddev),
        dual_temperature=jnp.mean(temperature),
        # Losses.
        loss_policy=jnp.mean(loss),
        loss_alpha=jnp.mean(loss_alpha_mean + loss_alpha_stddev),
        loss_temperature=jnp.mean(loss_temperature),
        # KL measurements.
        kl_q_rel=jnp.mean(kl_nonparametric) / epsilon,
        penalty_kl_q_rel=(
            (jnp.mean(penalty_kl_nonparametric) / epsilon_penalty) if action_penalization else None
        ),
        kl_mean_rel=jnp.mean(kl_mean, axis=0) / epsilon_mean,
        kl_stddev_rel=jnp.mean(kl_stddev, axis=0) / epsilon_stddev,
        # Q measurements.
        q_min=jnp.mean(jnp.min(target_sampled_q_values, axis=0)),
        q_max=jnp.mean(jnp.max(target_sampled_q_values, axis=0)),
        # If the policy has stddev, log summary stats for this as well.
        pi_stddev_min=jnp.mean(jnp.min(pi_stddev, axis=-1)),
        pi_stddev_max=jnp.mean(jnp.max(pi_stddev, axis=-1)),
        # Condition number of the diagonal covariance (actually, stddev) matrix.
        pi_stddev_cond=jnp.mean(jnp.max(pi_stddev, axis=-1) / jnp.min(pi_stddev, axis=-1)),
    )

    return loss, stats
