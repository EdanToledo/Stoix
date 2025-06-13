from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import rlax
from absl.testing import absltest, parameterized

from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation


class TruncatedGeneralizedAdvantageEstimationTest(parameterized.TestCase):
    """Test suite for truncated Generalized Advantage Estimation (GAE).

    This test suite validates the GAE implementation across various scenarios:
    - Standard GAE computation with different lambda values
    - Handling of episode truncation (continuing episodes)
    - Handling of episode termination (true episode ends)
    - Combined truncation and termination scenarios
    - Edge cases with multiple truncations

    Key concepts tested:
    - Truncation: Episode continues but needs to be cut for practical reasons
      (e.g., fixed rollout length). Bootstrap from value function.
    - Termination: Episode truly ends (e.g., game over). No bootstrapping.
    """

    def setUp(self) -> None:
        """Initialize test data for various test scenarios."""
        super().setUp()
        self._setup_basic_test_data()
        self._setup_expected_results()

    def _setup_basic_test_data(self) -> None:
        """Setup basic test data used across multiple tests."""
        # Basic 2-batch, 5-timestep example
        self.r_t = jnp.array(
            [
                [0.0, 0.0, 1.0, 0.0, -0.5],  # Batch 1: sparse rewards
                [0.0, 0.0, 0.0, 0.0, 1.0],  # Batch 2: final reward
            ]
        )

        # Values array includes initial state value and all timestep values
        self.values = jnp.array(
            [
                [1.0, 4.0, -3.0, -2.0, -1.0, -1.0],  # Batch 1: 6 values (T+1)
                [-3.0, -2.0, -1.0, 0.0, 5.0, -1.0],  # Batch 2: 6 values (T+1)
            ]
        )

        # Split values for v_tm1/v_t interface (for complex autoreset cases)
        self.v_tm1 = self.values[:, :-1]  # Current state values [B, T]
        self.v_t = self.values[:, 1:]  # Next state values for bootstrapping [B, T]

        # Discount factors (0 indicates termination)
        self.discount_t = jnp.array(
            [
                [0.99, 0.99, 0.99, 0.99, 0.99],  # Batch 1: no termination
                [0.9, 0.9, 0.9, 0.0, 0.9],  # Batch 2: termination at t=3
            ]
        )

        # Lambda values for testing array vs scalar
        self.array_lambda = jnp.full_like(self.discount_t, 0.9)

        # Truncation mask (1 indicates truncation point)
        self.truncation_t = jnp.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],  # Batch 1: no truncation
                [0.0, 0.0, 0.0, 1.0, 0.0],  # Batch 2: truncation at t=3
            ]
        )

    def _setup_expected_results(self) -> None:
        """Pre-computed expected GAE values for different lambda values.

        These values were computed manually using the GAE formula:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        self.expected = {
            1.0: np.array(
                [[-1.45118, -4.4557, 2.5396, 0.5249, -0.49], [3.0, 2.0, 1.0, 0.0, -4.9]],
                dtype=np.float32,
            ),
            0.7: np.array(
                [[-0.676979, -5.248167, 2.4846, 0.6704, -0.49], [2.2899, 1.73, 1.0, 0.0, -4.9]],
                dtype=np.float32,
            ),
            0.4: np.array(
                [[0.56731, -6.042, 2.3431, 0.815, -0.49], [1.725, 1.46, 1.0, 0.0, -4.9]],
                dtype=np.float32,
            ),
        }

    # ========== Helper Methods ==========

    def _compute_td_error(self, r: float, discount: float, v_next: float, v_curr: float) -> float:
        """Compute TD error: δ = r + γV(s') - V(s)"""
        return r + discount * v_next - v_curr

    def _assert_gae_shapes(
        self, advantages: jnp.ndarray, targets: jnp.ndarray, expected_shape: Tuple[int, ...]
    ) -> None:
        """Assert that GAE outputs have expected shapes."""
        chex.assert_shape(advantages, expected_shape)
        chex.assert_shape(targets, expected_shape)

    # ========== Core Functionality Tests ==========

    @chex.all_variants()
    @parameterized.named_parameters(("lambda_1.0", 1.0), ("lambda_0.7", 0.7), ("lambda_0.4", 0.4))
    def test_basic_gae_computation(self, lambda_: float) -> None:
        """Test basic GAE computation with different lambda values - No Truncation.

        Validates:
        - Correct advantage computation for various lambda (bias-variance tradeoff)
        - Correct target value computation (V + A)
        - Both interfaces (single values array vs separate v_tm1/v_t) produce same results
        """
        gae_fn = self.variant(batch_truncated_generalized_advantage_estimation)

        # Test with single values array interface
        advantages, targets = gae_fn(self.r_t, self.discount_t, lambda_, values=self.values)

        # Verify advantages match expected values
        np.testing.assert_allclose(advantages, self.expected[lambda_], atol=1e-3)

        # Verify targets = values + advantages
        expected_targets = self.expected[lambda_] + self.v_tm1
        np.testing.assert_allclose(targets, expected_targets, atol=1e-3)

        # Test with separate v_tm1/v_t interface (for complex autoreset cases)
        advantages_alt, targets_alt = gae_fn(
            self.r_t, self.discount_t, lambda_, v_tm1=self.v_tm1, v_t=self.v_t
        )

        # Both interfaces should produce identical results
        np.testing.assert_allclose(advantages, advantages_alt, atol=1e-6)
        np.testing.assert_allclose(targets, targets_alt, atol=1e-6)

        # Test rlax implementation for comparison
        rlax_advantages = jax.vmap(
            rlax.truncated_generalized_advantage_estimation, in_axes=(0, 0, None, 0)
        )(self.r_t, self.discount_t, lambda_, self.values)

        np.testing.assert_allclose(advantages, rlax_advantages, atol=1e-6)
        np.testing.assert_allclose(targets, self.values[:, :-1] + rlax_advantages, atol=1e-6)

    @chex.all_variants()
    def test_scalar_vs_array_lambda(self) -> None:
        """Test that scalar and array lambda produce identical results.

        This ensures the implementation correctly broadcasts scalar lambda values.
        """
        gae_fn = self.variant(batch_truncated_generalized_advantage_estimation)

        # Compute with scalar lambda
        scalar_adv, scalar_targets = gae_fn(
            self.r_t, self.discount_t, 0.9, v_tm1=self.v_tm1, v_t=self.v_t
        )

        # Compute with array lambda (same value everywhere)
        array_adv, array_targets = gae_fn(
            self.r_t, self.discount_t, self.array_lambda, v_tm1=self.v_tm1, v_t=self.v_t
        )

        np.testing.assert_allclose(scalar_adv, array_adv, atol=1e-6)
        np.testing.assert_allclose(scalar_targets, array_targets, atol=1e-6)

    # ========== Truncation Tests ==========

    @chex.all_variants()
    def test_truncation_vs_termination(self) -> None:
        """Test that truncation and termination are handled differently.

        Key difference:
        - Termination (discount=0): No bootstrapping, episode truly ends
        - Truncation (truncation_t=1): Bootstrap from value function, episode continues

        This test verifies that truncated episodes correctly bootstrap from the
        value function while terminated episodes do not.
        """
        # Test data: same rewards and values, but different ending conditions
        r_t = jnp.array([[0.0, 0.0, 0.0, 0.0]])
        values = jnp.array([[1.0, 1.0, 1.0, 1.0, 10.0]])  # High final value

        # Case 1: Truncation at t=2 (episode continues, should bootstrap)
        discount_truncation = jnp.array([[0.9, 0.9, 0.9, 0.9]])
        truncation_mask = jnp.array([[0.0, 0.0, 1.0, 0.0]])

        # Case 2: Termination at t=2 (episode ends, no bootstrap)
        discount_termination = jnp.array([[0.9, 0.9, 0.0, 0.9]])

        gae_fn = self.variant(batch_truncated_generalized_advantage_estimation)

        # Compute advantages for truncation case
        trunc_adv, _ = gae_fn(
            r_t,
            discount_truncation,
            1.0,
            v_tm1=values[:, :-1],
            v_t=values[:, 1:],
            truncation_t=truncation_mask,
        )

        # Compute advantages for termination case
        term_adv, _ = gae_fn(
            r_t, discount_termination, 1.0, v_tm1=values[:, :-1], v_t=values[:, 1:]
        )

        # At t=2, truncation should bootstrap (include future value)
        # while termination should not
        # TD errors at t=2:
        # Truncation: δ = 0 + 0.9 * 1 - 1 = -0.1
        # Termination: δ = 0 + 0.0 * 1 - 1 = -1.0
        np.testing.assert_allclose(trunc_adv[0, 2], -0.1, atol=1e-5)
        np.testing.assert_allclose(term_adv[0, 2], -1.0, atol=1e-5)

        # Earlier timesteps should also differ due to different propagation
        self.assertFalse(np.allclose(trunc_adv[0, :2], term_adv[0, :2], atol=1e-5))

    @chex.all_variants()
    def test_multiple_truncations(self) -> None:
        """Test GAE with multiple truncation points in a single sequence.

        This validates that the backward scan correctly resets advantage
        accumulation at each truncation point while preserving the TD error
        at the truncation point itself.
        """
        # Single sequence with truncations at t=2 and t=4
        r_t = jnp.array([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        values = jnp.array([[0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0]])
        discount_t = jnp.full((1, 7), 0.9)
        truncation_t = jnp.array([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]])

        gae_fn = self.variant(batch_truncated_generalized_advantage_estimation)
        advantages, _ = gae_fn(r_t, discount_t, 1.0, values=values, truncation_t=truncation_t)

        # Manually compute expected advantages
        # Working backwards from t=6 to t=0

        # t=6: δ_6 = 0 + 0.9*0 - 0 = 0
        self.assertAlmostEqual(advantages[0, 6], 0.0, places=3)

        # t=5: δ_5 = 1 + 0.9*0 - 1 = 0, A_5 = 0 + 0.9*0 = 0
        self.assertAlmostEqual(advantages[0, 5], 0.0, places=3)

        # t=4 (truncation): δ_4 = 0 + 0.9*1 - 0 = 0.9, A_4 = 0.9 (no accumulation)
        self.assertAlmostEqual(advantages[0, 4], 0.9, places=3)

        # t=3: δ_3 = 0 + 0.9*0 - 1 = -1, A_3 = -1 + 0.9*0.9 = -0.19
        self.assertAlmostEqual(advantages[0, 3], -0.19, places=2)

        # t=2 (truncation): δ_2 = 0 + 0.9*1 - 2 = -1.1, A_2 = -1.1 (no accumulation)
        self.assertAlmostEqual(advantages[0, 2], -1.1, places=3)

    # ========== Complex Scenario Tests ==========

    @chex.all_variants()
    def test_mixed_truncation_and_termination(self) -> None:
        """Test GAE with mixed truncation and termination across batches.

        This comprehensive test validates correct handling when different
        episodes in a batch have different ending conditions:
        - Episode 1: Normal termination (game over)
        - Episode 2: Truncation (rollout limit reached)
        - Episode 3: Normal termination (game over)
        """
        # Three episodes with different characteristics
        r_t = jnp.array(
            [
                [1.0, 0.0, 0.0],  # Episode 1: early reward
                [0.0, 0.5, 0.0],  # Episode 2: middle reward
                [0.0, 0.0, 2.0],  # Episode 3: final reward
            ]
        )

        values = jnp.array(
            [
                [1.0, 1.0, 0.0, 0.0],  # Episode 1 values
                [1.0, 1.0, 1.0, 0.0],  # Episode 2 values (continuing)
                [1.0, 1.0, 0.5, 0.0],  # Episode 3 values
            ]
        )

        # Episode 1 & 3 terminate (discount=0), Episode 2 continues
        discount_t = jnp.array(
            [
                [0.9, 0.9, 0.0],  # Episode 1: terminal at t=2
                [0.9, 0.9, 0.9],  # Episode 2: no termination
                [0.9, 0.9, 0.0],  # Episode 3: terminal at t=2
            ]
        )

        # Only Episode 2 is truncated
        truncation_t = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Episode 1: no truncation
                [0.0, 0.0, 1.0],  # Episode 2: truncated at t=2
                [0.0, 0.0, 0.0],  # Episode 3: no truncation
            ]
        )

        gae_fn = self.variant(batch_truncated_generalized_advantage_estimation)
        advantages, targets = gae_fn(r_t, discount_t, 1.0, values=values, truncation_t=truncation_t)

        # Verify key differences:
        # Episode 1 (terminated): Should not bootstrap at t=2
        # δ_2 = 0 + 0*0 - 0 = 0
        self.assertAlmostEqual(advantages[0, 2], 0.0, places=3)

        # Episode 2 (truncated): Should bootstrap at t=2
        # δ_2 = 0 + 0.9*0 - 1 = -1
        self.assertAlmostEqual(advantages[1, 2], -1.0, places=3)

        # Episode 3 (terminated): Should not bootstrap at t=2
        # δ_2 = 2 + 0*0 - 0.5 = 1.5
        self.assertAlmostEqual(advantages[2, 2], 1.5, places=3)

        # Verify targets are computed correctly
        np.testing.assert_allclose(targets, values[:, :-1] + advantages, atol=1e-3)

    # ========== Interface Difference Tests ==========

    @chex.all_variants()
    def test_values_vs_v_tm1_v_t_interface(self) -> None:
        """Test the difference between values and v_tm1/v_t interfaces.

        The key difference is in how they handle autoreset scenarios:
        - values interface: Simple, assumes continuous sequence [0, T]
        - v_tm1/v_t interface: For complex autoreset where episodes restart

        In autoreset with truncation:
        - v_tm1: Skips final timestep values (sequences like [0,1,2,3,0,1,2,3,...])
        - v_t: Skips initial timestep but includes final (sequences like [1,2,3,4,1,2,3,4,...])

        This allows proper bootstrapping when episode is cut and immediately restarts.
        """
        # Scenario: 2 episodes of length 3, truncated and autoreset
        # Episode 1: steps 0,1,2 then truncated and reset
        # Episode 2: steps 0,1,2 (continuing)

        r_t = jnp.array([[1.0, 0.0, 0.0, 0.5, 0.0, 0.0]])  # 6 timesteps
        discount_t = jnp.full((1, 6), 0.9)

        # For autoreset scenario with truncation at t=2 and at the end
        truncation_t = jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]])

        # v_tm1/v_t interface for autoreset:
        # v_tm1: values at current states, skipping episode boundaries
        # Represents: [V(s0_ep1), V(s1_ep1), V(s2_ep1), V(s0_ep2), V(s1_ep2), V(s2_ep2)]
        v_tm1_autoreset = jnp.array([[1.0, 2.0, 3.0, 1.0, 1.5, 2.0]])

        # v_t: values for bootstrapping, includes proper values at truncation
        # At truncation (t=2), we bootstrap from initial value of next episode
        # Represents: [V(s1_ep1), V(s2_ep1), V(s3_ep1)=T, V(s1_ep2), V(s2_ep2), V(s3_ep2)=T]
        v_t_autoreset = jnp.array(
            [[2.0, 3.0, 4.0, 1.5, 2.0, 1.0]]
        )  # Note: V(s0_ep2)=1.0 at position 2

        # Simple values interface (would be incorrect for autoreset):
        # Just includes all values sequentially without special handling
        values_simple = jnp.array([[1.0, 2.0, 3.0, 1.0, 1.5, 2.0, 100.0]])
        # Represents: [V(s0_ep1), V(s1_ep1), V(s2_ep1), V(s0_ep2), V(s1_ep2), V(s2_ep2), V(s0_ep3)]
        gae_fn = self.variant(batch_truncated_generalized_advantage_estimation)

        # Compute with autoreset interface
        adv_autoreset, _ = gae_fn(
            r_t,
            discount_t,
            1.0,
            v_tm1=v_tm1_autoreset,
            v_t=v_t_autoreset,
            truncation_t=truncation_t,
        )

        # Compute with simple values interface (no truncation specified)
        adv_simple, _ = gae_fn(
            r_t, discount_t, 1.0, values=values_simple, truncation_t=truncation_t
        )

        # At truncation point (t=2), the TD errors differ:
        # Autoreset: δ_2 = 0 + 0.9*4.0 - 3.0 = 0.6 (bootstraps from V(sT_ep1))
        # Simple: δ_2 = 0 + 0.9*1.0 - 3.0 = -2.1 (bootstraps from V(s0_ep2) - due to
        # truncation this would be incorrect)

        # The key difference is that autoreset correctly handles the episode boundary truncation
        # while simple interface would incorrectly propagate advantages across episodes

        # Check that with truncation, advantage at t=2 is just the TD error
        expected_td_at_truncation = (
            r_t[0, 2] + discount_t[0, 2] * v_t_autoreset[0, 2] - v_tm1_autoreset[0, 2]
        )
        np.testing.assert_allclose(adv_autoreset[0, 2], expected_td_at_truncation, atol=1e-3)
        np.testing.assert_allclose(adv_autoreset[0, 2], 0.9 * 4.0 - 3.0, atol=1e-3)

        expected_incorrect_td = (
            r_t[0, 2] + discount_t[0, 2] * values_simple[0, 3] - values_simple[0, 2]
        )
        np.testing.assert_allclose(adv_simple[0, 2], expected_incorrect_td, atol=1e-3)
        np.testing.assert_allclose(adv_simple[0, 2], 0.9 * 1.0 - 3.0, atol=1e-3)

        expected_td_at_truncation = (
            r_t[0, -1] + discount_t[0, -1] * v_t_autoreset[0, -1] - v_tm1_autoreset[0, -1]
        )
        np.testing.assert_allclose(adv_autoreset[0, -1], expected_td_at_truncation, atol=1e-3)
        np.testing.assert_allclose(adv_autoreset[0, -1], 0.9 * 1.0 - 2.0, atol=1e-3)

        expected_incorrect_td = (
            r_t[0, -1] + discount_t[0, -1] * values_simple[0, -1] - values_simple[0, -2]
        )
        np.testing.assert_allclose(adv_simple[0, -1], expected_incorrect_td, atol=1e-3)
        np.testing.assert_allclose(adv_simple[0, -1], 0.9 * 100 - 2, atol=1e-3)

        # Without truncation handling, advantages would propagate incorrectly
        # The simple interface would treat this as one continuous episode
        self.assertFalse(np.allclose(adv_autoreset, adv_simple, atol=1e-3))

        # verify that it is the same if we remove truncation and use termination in
        # the same position
        discount_t = 1 - truncation_t
        adv_autoreset_no_trunc, _ = gae_fn(
            r_t, discount_t, 1.0, v_tm1=v_tm1_autoreset, v_t=v_t_autoreset
        )
        adv_simple_no_trunc, _ = gae_fn(r_t, discount_t, 1.0, values=values_simple)
        np.testing.assert_allclose(adv_autoreset_no_trunc, adv_simple_no_trunc, atol=1e-6)

    @chex.all_variants()
    def test_autoreset_with_initial_values(self) -> None:
        """Test autoreset scenario where episodes have different initial values.

        This demonstrates why the v_tm1/v_t interface is necessary:
        when episodes are truncated and immediately reset to different initial states.
        """
        # Two episodes with different initial state values
        # Episode 1: V(s0) = 5.0 (high value initial state)
        # Episode 2: V(s0) = 1.0 (low value initial state)

        r_t = jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0]])  # 5 timesteps
        discount_t = jnp.full((1, 5), 0.9)
        truncation_t = jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0]])  # Truncate after step 2

        # Episode 1 values: [5.0, 4.0, 3.0] (decreasing)
        # Episode 2 values: [1.0, 2.0] (increasing)
        v_tm1 = jnp.array([[5.0, 4.0, 3.0, 1.0, 2.0]])

        # For v_t at truncation point, we need V(s0) of episode 2
        v_t = jnp.array([[4.0, 3.0, 1.0, 2.0, 0.0]])  # Note: 1.0 is V(s0_ep2)

        gae_fn = self.variant(batch_truncated_generalized_advantage_estimation)

        advantages, _ = gae_fn(
            r_t, discount_t, 1.0, v_tm1=v_tm1, v_t=v_t, truncation_t=truncation_t
        )

        # At truncation (t=2):
        # δ_2 = 1.0 + 0.9*1.0 - 3.0 = -1.1
        # This correctly uses V(s0) of the next episode for bootstrapping
        # But crucially, A_2 = δ_2 only (no future accumulation due to truncation)
        expected_td = 1.0 + 0.9 * 1.0 - 3.0
        np.testing.assert_allclose(advantages[0, 2], expected_td, atol=1e-3)

        # After truncation (t=3), we're in a new episode
        # But we need to account for the future advantage from t=4
        # δ_3 = 0.0 + 0.9*2.0 - 1.0 = 0.8
        # δ_4 = 0.0 + 0.9*0.0 - 2.0 = -2.0
        # A_4 = -2.0 (last timestep)
        # A_3 = δ_3 + γλ*A_4 = 0.8 + 0.9*1.0*(-2.0) = 0.8 - 1.8 = -1.0
        expected_adv_t3 = -1.0
        np.testing.assert_allclose(advantages[0, 3], expected_adv_t3, atol=1e-3)

    # ========== Edge Case Tests ==========

    @chex.all_variants()
    def test_all_truncated(self) -> None:
        """Test edge case where every timestep is truncated.

        In this case, advantages should just be TD errors with no
        temporal credit assignment.
        """
        r_t = jnp.array([[1.0, 0.5, -0.5]])
        values = jnp.array([[1.0, 2.0, 1.5, 1.0]])
        discount_t = jnp.full((1, 3), 0.9)
        truncation_t = jnp.ones((1, 3))  # All timesteps truncated

        gae_fn = self.variant(batch_truncated_generalized_advantage_estimation)
        advantages, _ = gae_fn(r_t, discount_t, 1.0, values=values, truncation_t=truncation_t)

        # Each advantage should just be its TD error
        for t in range(3):
            expected_td = self._compute_td_error(
                r_t[0, t], discount_t[0, t], values[0, t + 1], values[0, t]
            )
            self.assertAlmostEqual(advantages[0, t], expected_td, places=3)

    def test_time_major_format(self) -> None:
        """Test that time_major format produces correct results."""
        # Use the non-JIT version directly
        gae_fn = batch_truncated_generalized_advantage_estimation

        # Compute with batch-major (default)
        batch_major_adv, batch_major_targets = gae_fn(
            self.r_t, self.discount_t, 1.0, values=self.values, time_major=False
        )

        # Transpose inputs for time-major
        r_t_tm = jnp.transpose(self.r_t, (1, 0))
        discount_t_tm = jnp.transpose(self.discount_t, (1, 0))
        values_tm = jnp.transpose(self.values, (1, 0))

        # Compute with time-major
        time_major_adv, time_major_targets = gae_fn(
            r_t_tm, discount_t_tm, 1.0, values=values_tm, time_major=True
        )

        # Results should match after transposing back
        np.testing.assert_allclose(
            batch_major_adv, jnp.transpose(time_major_adv, (1, 0)), atol=1e-6
        )
        np.testing.assert_allclose(
            batch_major_targets, jnp.transpose(time_major_targets, (1, 0)), atol=1e-6
        )


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
