from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp

# These functions are generally taken from rlax but edited to explicitly take in a batch of data.
# This is because the original rlax functions are not batched and are meant to be used with vmap,
# which can be much slower.


def batch_truncated_generalized_advantage_estimation(
    r_t: chex.Array,
    discount_t: chex.Array,
    lambda_: Union[chex.Array, chex.Scalar],
    values: chex.Array,
    stop_target_gradients: bool = True,
    time_major: bool = False,
    standardize_advantages: bool = False,
    truncation_flags: Optional[chex.Array] = None,
) -> Tuple[chex.Array, chex.Array]:
    """Computes truncated generalized advantage estimates for a sequence length k.

    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ₊₁ + γₜ₊₁ * v(sₜ₊₁) - v(sₜ).

    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Note: This paper uses a different notation than the RLax standard
    convention that follows Sutton & Barto. We use rₜ₊₁ to denote the reward
    received after acting in state sₜ, while the PPO paper uses rₜ.

    Args:
        r_t: Sequence of rewards at times [1, k]
        discount_t: Sequence of discounts at times [1, k]
        lambda_: Mixing parameter; a scalar or sequence of lambda_t at times [1, k]
        values: Sequence of values under π at times [0, k]
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
        time_major: If True, the first dimension of the input tensors is the time
        dimension.
        standardize_advantages: If True, standardize the advantages.
        truncation_flags: Optional sequence of truncation flags at times [1, k].

    Returns:
        Multistep truncated generalized advantage estimation at times [0, k-1].
        The target values at times [0, k-1] are also returned.
    """

    if truncation_flags is None:
        truncation_flags = jnp.zeros_like(r_t)

    truncation_mask = 1.0 - truncation_flags

    # Swap axes to make time axis the first dimension
    if not time_major:
        batch_size = r_t.shape[0]
        r_t, discount_t, values, truncation_mask = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (r_t, discount_t, values, truncation_mask)
        )
    else:
        batch_size = r_t.shape[1]

    chex.assert_type([r_t, values, discount_t, truncation_mask], float)

    lambda_ = jnp.ones_like(discount_t) * lambda_  # If scalar, make into vector.

    delta_t = r_t + discount_t * values[1:] - values[:-1]
    delta_t *= truncation_mask

    # Iterate backwards to calculate advantages.
    def _body(
        acc: chex.Array, xs: Tuple[chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array]:
        deltas, discounts, lambda_, trunc_mask = xs
        acc = deltas + discounts * lambda_ * trunc_mask * acc
        return acc, acc

    _, advantage_t = jax.lax.scan(
        _body,
        jnp.zeros(batch_size),
        (delta_t, discount_t, lambda_, truncation_mask),
        reverse=True,
        unroll=16,
    )

    target_values = values[:-1] + advantage_t
    advantage_t *= truncation_mask

    if not time_major:
        # Swap axes back to original shape
        advantage_t, target_values = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (advantage_t, target_values)
        )

    if stop_target_gradients:
        advantage_t, target_values = jax.tree_util.tree_map(
            lambda x: jax.lax.stop_gradient(x), (advantage_t, target_values)
        )

    if standardize_advantages:
        advantage_t = jax.nn.standardize(advantage_t, axis=(0, 1))

    return advantage_t, target_values


def batch_n_step_bootstrapped_returns(
    r_t: chex.Array,
    discount_t: chex.Array,
    v_t: chex.Array,
    n: int,
    lambda_t: float = 1.0,
    stop_target_gradients: bool = True,
) -> chex.Array:
    """Computes strided n-step bootstrapped return targets over a batch of sequences.

    The returns are computed according to the below equation iterated `n` times:

        Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].

    When lambda_t == 1. (default), this reduces to

        Gₜ = rₜ₊₁ + γₜ₊₁ * (rₜ₊₂ + γₜ₊₂ * (... * (rₜ₊ₙ + γₜ₊ₙ * vₜ₊ₙ ))).

    Args:
        r_t: rewards at times B x [1, ..., T].
        discount_t: discounts at times B x [1, ..., T].
        v_t: state or state-action values to bootstrap from at time B x [1, ...., T].
        n: number of steps over which to accumulate reward before bootstrapping.
        lambda_t: lambdas at times B x [1, ..., T]. Shape is [], or B x [T-1].
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.

    Returns:
        estimated bootstrapped returns at times B x [0, ...., T-1]
    """
    # swap axes to make time axis the first dimension
    r_t, discount_t, v_t = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), (r_t, discount_t, v_t)
    )
    seq_len = r_t.shape[0]
    batch_size = r_t.shape[1]

    # Maybe change scalar lambda to an array.
    lambda_t = jnp.ones_like(discount_t) * lambda_t

    # Shift bootstrap values by n and pad end of sequence with last value v_t[-1].
    pad_size = min(n - 1, seq_len)
    targets = jnp.concatenate([v_t[n - 1 :], jnp.array([v_t[-1]] * pad_size)], axis=0)

    # Pad sequences. Shape is now (T + n - 1,).
    r_t = jnp.concatenate([r_t, jnp.zeros((n - 1, batch_size))], axis=0)
    discount_t = jnp.concatenate([discount_t, jnp.ones((n - 1, batch_size))], axis=0)
    lambda_t = jnp.concatenate([lambda_t, jnp.ones((n - 1, batch_size))], axis=0)
    v_t = jnp.concatenate([v_t, jnp.array([v_t[-1]] * (n - 1))], axis=0)

    # Work backwards to compute n-step returns.
    for i in reversed(range(n)):
        r_ = r_t[i : i + seq_len]
        discount_ = discount_t[i : i + seq_len]
        lambda_ = lambda_t[i : i + seq_len]
        v_ = v_t[i : i + seq_len]
        targets = r_ + discount_ * ((1.0 - lambda_) * v_ + lambda_ * targets)

    targets = jnp.swapaxes(targets, 0, 1)
    return jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(targets), targets)


def batch_general_off_policy_returns_from_q_and_v(
    q_t: chex.Array,
    v_t: chex.Array,
    r_t: chex.Array,
    discount_t: chex.Array,
    c_t: chex.Array,
    stop_target_gradients: bool = False,
) -> chex.Array:
    """Calculates targets for various off-policy evaluation algorithms.

    Given a window of experience of length `K+1`, generated by a behaviour policy
    μ, for each time-step `t` we can estimate the return `G_t` from that step
    onwards, under some target policy π, using the rewards in the trajectory, the
    values under π of states and actions selected by μ, according to equation:

      Gₜ = rₜ₊₁ + γₜ₊₁ * (vₜ₊₁ - cₜ₊₁ * q(aₜ₊₁) + cₜ₊₁* Gₜ₊₁),

    where, depending on the choice of `c_t`, the algorithm implements:

      Importance Sampling             c_t = π(x_t, a_t) / μ(x_t, a_t),
      Harutyunyan's et al. Q(lambda)  c_t = λ,
      Precup's et al. Tree-Backup     c_t = π(x_t, a_t),
      Munos' et al. Retrace           c_t = λ min(1, π(x_t, a_t) / μ(x_t, a_t)).

    See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
    (https://arxiv.org/abs/1606.02647).

    Args:
      q_t: Q-values under π of actions executed by μ at times [1, ..., K - 1].
      v_t: Values under π at times [1, ..., K].
      r_t: rewards at times [1, ..., K].
      discount_t: discounts at times [1, ..., K].
      c_t: weights at times [1, ..., K - 1].
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.

    Returns:
      Off-policy estimates of the generalized returns from states visited at times
      [0, ..., K - 1].
    """
    q_t, v_t, r_t, discount_t, c_t = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), (q_t, v_t, r_t, discount_t, c_t)
    )

    g = r_t[-1] + discount_t[-1] * v_t[-1]  # G_K-1.

    def _body(
        acc: chex.Array, xs: Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array]:
        reward, discount, c, v, q = xs
        acc = reward + discount * (v - c * q + c * acc)
        return acc, acc

    _, returns = jax.lax.scan(
        _body, g, (r_t[:-1], discount_t[:-1], c_t, v_t[:-1], q_t), reverse=True
    )
    returns = jnp.concatenate([returns, g[jnp.newaxis]], axis=0)

    returns = jnp.swapaxes(returns, 0, 1)
    return jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(returns), returns)


def batch_retrace_continuous(
    q_tm1: chex.Array,
    q_t: chex.Array,
    v_t: chex.Array,
    r_t: chex.Array,
    discount_t: chex.Array,
    log_rhos: chex.Array,
    lambda_: Union[chex.Array, float],
    stop_target_gradients: bool = True,
) -> chex.Array:
    """Retrace continuous.

    See "Safe and Efficient Off-Policy Reinforcement Learning" by Munos et al.
    (https://arxiv.org/abs/1606.02647).

    Args:
      q_tm1: Q-values at times [0, ..., K - 1].
      q_t: Q-values evaluated at actions collected using behavior
        policy at times [1, ..., K - 1].
      v_t: Value estimates of the target policy at times [1, ..., K].
      r_t: reward at times [1, ..., K].
      discount_t: discount at times [1, ..., K].
      log_rhos: Log importance weight pi_target/pi_behavior evaluated at actions
        collected using behavior policy [1, ..., K - 1].
      lambda_: scalar or a vector of mixing parameter lambda.
      stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.

    Returns:
      Retrace error.
    """

    c_t = jnp.minimum(1.0, jnp.exp(log_rhos)) * lambda_

    # The generalized returns are independent of Q-values and cs at the final
    # state.
    target_tm1 = batch_general_off_policy_returns_from_q_and_v(q_t, v_t, r_t, discount_t, c_t)

    target_tm1 = jax.lax.select(
        stop_target_gradients, jax.lax.stop_gradient(target_tm1), target_tm1
    )
    return target_tm1 - q_tm1


def batch_lambda_returns(
    r_t: chex.Array,
    discount_t: chex.Array,
    v_t: chex.Array,
    lambda_: chex.Numeric = 1.0,
    stop_target_gradients: bool = False,
    time_major: bool = False,
) -> chex.Array:
    """Estimates a multistep truncated lambda return from a trajectory.

    Given a a trajectory of length `T+1`, generated under some policy π, for each
    time-step `t` we can estimate a target return `G_t`, by combining rewards,
    discounts, and state values, according to a mixing parameter `lambda`.

    The parameter `lambda_`  mixes the different multi-step bootstrapped returns,
    corresponding to accumulating `k` rewards and then bootstrapping using `v_t`.

        rₜ₊₁ + γₜ₊₁ vₜ₊₁
        rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ vₜ₊₂
        rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ rₜ₊₂ + γₜ₊₁ γₜ₊₂ γₜ₊₃ vₜ₊₃

    The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:

        Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].

    In the `on-policy` case, we estimate a return target `G_t` for the same
    policy π that was used to generate the trajectory. In this setting the
    parameter `lambda_` is typically a fixed scalar factor. Depending
    on how values `v_t` are computed, this function can be used to construct
    targets for different multistep reinforcement learning updates:

        TD(λ):  `v_t` contains the state value estimates for each state under π.
        Q(λ):  `v_t = max(q_t, axis=-1)`, where `q_t` estimates the action values.
        Sarsa(λ):  `v_t = q_t[..., a_t]`, where `q_t` estimates the action values.

    In the `off-policy` case, the mixing factor is a function of state, and
    different definitions of `lambda` implement different off-policy corrections:

        Per-decision importance sampling:  λₜ = λ ρₜ = λ [π(aₜ|sₜ) / μ(aₜ|sₜ)]
        V-trace, as instantiated in IMPALA:  λₜ = min(1, ρₜ)

    Note that the second option is equivalent to applying per-decision importance
    sampling, but using an adaptive λ(ρₜ) = min(1/ρₜ, 1), such that the effective
    bootstrap parameter at time t becomes λₜ = λ(ρₜ) * ρₜ = min(1, ρₜ).
    This is the interpretation used in the ABQ(ζ) algorithm (Mahmood 2017).

    Of course this can be augmented to include an additional factor λ.  For
    instance we could use V-trace with a fixed additional parameter λ = 0.9, by
    setting λₜ = 0.9 * min(1, ρₜ) or, alternatively (but not equivalently),
    λₜ = min(0.9, ρₜ).

    Estimated return are then often used to define a td error, e.g.:  ρₜ(Gₜ - vₜ).

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/sutton/book/ebook/node74.html).

    Args:
        r_t: sequence of rewards rₜ for timesteps t in B x [1, T].
        discount_t: sequence of discounts γₜ for timesteps t in B x [1, T].
        v_t: sequence of state values estimates under π for timesteps t in B x [1, T].
        lambda_: mixing parameter; a scalar or a vector for timesteps t in B x [1, T].
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.
        time_major: If True, the first dimension of the input tensors is the time
        dimension.

    Returns:
        Multistep lambda returns.
    """

    chex.assert_rank([r_t, discount_t, v_t, lambda_], [2, 2, 2, {0, 1, 2}])
    chex.assert_type([r_t, discount_t, v_t, lambda_], float)
    chex.assert_equal_shape([r_t, discount_t, v_t])

    # Swap axes to make time axis the first dimension
    if not time_major:
        r_t, discount_t, v_t = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (r_t, discount_t, v_t)
        )

    # If scalar make into vector.
    lambda_ = jnp.ones_like(discount_t) * lambda_

    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    def _body(
        acc: chex.Array, xs: Tuple[chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array]:
        returns, discounts, values, lambda_ = xs
        acc = returns + discounts * ((1 - lambda_) * values + lambda_ * acc)
        return acc, acc

    _, returns = jax.lax.scan(_body, v_t[-1], (r_t, discount_t, v_t, lambda_), reverse=True)

    if not time_major:
        returns = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), returns)

    return jax.lax.select(stop_target_gradients, jax.lax.stop_gradient(returns), returns)


def batch_discounted_returns(
    r_t: chex.Array,
    discount_t: chex.Array,
    v_t: chex.Array,
    stop_target_gradients: bool = False,
    time_major: bool = False,
) -> chex.Array:
    """Calculates a discounted return from a trajectory.

    The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:

        Gₜ = rₜ₊₁ + γₜ₊₁ Gₜ₊₁.

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/sutton/book/ebook/node61.html).

    Args:
        r_t: reward sequence at time t.
        discount_t: discount sequence at time t.
        v_t: value sequence or scalar at time t.
        stop_target_gradients: bool indicating whether or not to apply stop gradient
        to targets.

    Returns:
        Discounted returns.
    """
    chex.assert_rank([r_t, discount_t, v_t], [2, 2, {0, 1, 2}])
    chex.assert_type([r_t, discount_t, v_t], float)

    # If scalar make into vector.
    bootstrapped_v = jnp.ones_like(discount_t) * v_t
    return batch_lambda_returns(
        r_t,
        discount_t,
        bootstrapped_v,
        lambda_=1.0,
        stop_target_gradients=stop_target_gradients,
        time_major=time_major,
    )
