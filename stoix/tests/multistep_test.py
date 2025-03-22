import jax
import jax.numpy as jnp
import chex
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation, importance_corrected_td_errors

class TruncatedGeneralizedAdvantageEstimationTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()

        self.r_t = jnp.array([[0.0, 0.0, 1.0, 0.0, -0.5], [0.0, 0.0, 0.0, 0.0, 1.0]])
        self.v_t = jnp.array(
            [[1.0, 4.0, -3.0, -2.0, -1.0, -1.0], [-3.0, -2.0, -1.0, 0.0, 5.0, -1.0]]
        )
        self.discount_t = jnp.array(
            [[0.99, 0.99, 0.99, 0.99, 0.99], [0.9, 0.9, 0.9, 0.0, 0.9]]
        )
        self.dummy_rho_tm1 = jnp.array(
            [[1.0, 1.0, 1.0, 1.0, 1], [1.0, 1.0, 1.0, 1.0, 1.0]]
        )
        self.array_lambda = jnp.array(
            [[0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.9]]
        )
        self.truncation_t = jnp.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]
        )

        # Different expected results for different values of lambda.
        self.expected = {}
        self.expected[1.0] = np.array(
            [[-1.45118, -4.4557, 2.5396, 0.5249, -0.49], [3.0, 2.0, 1.0, 0.0, -4.9]],
            dtype=np.float32,
        )
        self.expected[0.7] = np.array(
            [
                [-0.676979, -5.248167, 2.4846, 0.6704, -0.49],
                [2.2899, 1.73, 1.0, 0.0, -4.9],
            ],
            dtype=np.float32,
        )
        self.expected[0.4] = np.array(
            [[0.56731, -6.042, 2.3431, 0.815, -0.49], [1.725, 1.46, 1.0, 0.0, -4.9]],
            dtype=np.float32,
        )

        # Expected results for truncation case
        self.truncation_expected = np.array(
            [
                [-1.45118, -4.4557, 2.5396, 0.5249, -0.49],
                [3.0, 2.0, 1.0, 0.0, -4.9],
            ],  # Same result because truncation is at t=3 which already has discount=0
            dtype=np.float32,
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        ("lambda1", 1.0), ("lambda0.7", 0.7), ("lambda0.4", 0.4)
    )
    def test_truncated_gae(self, lambda_):
        """Tests truncated GAE for a full batch."""
        batched_advantage_fn_variant = self.variant(
            batch_truncated_generalized_advantage_estimation
        )
        advantages, targets = batched_advantage_fn_variant(
            self.r_t, self.discount_t, lambda_, self.v_t, None
        )
        np.testing.assert_allclose(self.expected[lambda_], advantages, atol=1e-3)
        # Check that targets are advantages + values
        expected_targets = self.expected[lambda_] + self.v_t[:, :-1]
        np.testing.assert_allclose(expected_targets, targets, atol=1e-3)

    @chex.all_variants()
    def test_array_lambda(self):
        """Tests that truncated GAE is consistent with scalar or array lambda_."""
        scalar_lambda_fn = self.variant(batch_truncated_generalized_advantage_estimation)
        array_lambda_fn = self.variant(batch_truncated_generalized_advantage_estimation)
        scalar_lambda_advantages, scalar_lambda_targets = scalar_lambda_fn(
            self.r_t, self.discount_t, 0.9, self.v_t, None
        )
        array_lambda_advantages, array_lambda_targets = array_lambda_fn(
            self.r_t, self.discount_t, self.array_lambda, self.v_t, None
        )
        np.testing.assert_allclose(
            scalar_lambda_advantages, array_lambda_advantages, atol=1e-3
        )
        np.testing.assert_allclose(
            scalar_lambda_targets, array_lambda_targets, atol=1e-3
        )

    @chex.all_variants()
    @parameterized.named_parameters(
        ("lambda_1", 1.0), ("lambda_0.7", 0.7), ("lambda_0.4", 0.4)
    )
    def test_truncation_with_different_lambdas(self, lambda_):
        """Tests truncated GAE with truncation points using different lambda values."""
        # Create a version of discount_t without the zero at the truncation point
        modified_discount_t = jnp.array(
            [[0.99, 0.99, 0.99, 0.99, 0.99], [0.9, 0.9, 0.9, 0.9, 0.9]]
        )

        # Test with truncation
        batched_advantage_fn_variant = self.variant(
            batch_truncated_generalized_advantage_estimation
        )
        advantages, targets = batched_advantage_fn_variant(
            self.r_t, modified_discount_t, lambda_, self.v_t, truncation_t = self.truncation_t
        )

        # For the first batch, there is no truncation so results should match the standard GAE
        batched_advantage_fn_no_truncation = self.variant(
            batch_truncated_generalized_advantage_estimation
        )
        standard_advantages, standard_targets = batched_advantage_fn_no_truncation(
            self.r_t, modified_discount_t, lambda_, self.v_t, None
        )

        # First row should match (no truncation)
        np.testing.assert_allclose(advantages[0], standard_advantages[0], atol=1e-3)
        np.testing.assert_allclose(targets[0], standard_targets[0], atol=1e-3)

        # Second row should differ where truncation affects it
        # Let's manually verify the last timestep (t=4) should match since it's after truncation
        np.testing.assert_allclose(
            advantages[1, 4], standard_advantages[1, 4], atol=1e-3
        )
        np.testing.assert_allclose(targets[1, 4], standard_targets[1, 4], atol=1e-3)

        # The truncation point itself (t=3) should just have its TD error
        v_t = self.v_t[1]  # Values for second batch
        r_t = self.r_t[1]  # Rewards for second batch
        discount_t = modified_discount_t[1]  # Using the non-zero discount
        expected_t3 = r_t[3] + discount_t[3] * v_t[4] - v_t[3]  # 0 + 0.9*5 - 0 = 4.5
        np.testing.assert_allclose(advantages[1, 3], expected_t3, atol=1e-3)
        # Target at truncation point should be value + advantage
        np.testing.assert_allclose(targets[1, 3], v_t[3] + expected_t3, atol=1e-3)

    @chex.all_variants()
    def test_multiple_truncations(self):
        """Tests GAE with multiple truncation points in a sequence."""
        # Create a test case with multiple truncations
        r_t = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        v_t = jnp.array(
            [0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        )  # One more value than rewards
        discount_t = jnp.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        truncation_t = jnp.array(
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        )  # Truncations at t=2 and t=4

        r_t, discount_t, v_t, truncation_t = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), (r_t, discount_t, v_t, truncation_t)
        )

        # Compute GAE with truncations
        advantage_fn_variant = self.variant(batch_truncated_generalized_advantage_estimation)
        advantages, targets = advantage_fn_variant(
            r_t, discount_t, 1.0, v_t, truncation_t=truncation_t
        )

        advantages, targets, r_t, discount_t, v_t, truncation_t = (
            jax.tree_util.tree_map(
                lambda x: jnp.squeeze(x, axis=0),
                (advantages, targets, r_t, discount_t, v_t, truncation_t),
            )
        )

        # Calculate the expected values manually
        # For t=6 (last step), the advantage is just the TD error
        expected_t6 = r_t[6] + discount_t[6] * v_t[7] - v_t[6]  # 0 + 0.9*0 - 0 = 0

        # For t=5, advantage includes t=5's TD error and discounted advantage from t=6
        delta_t5 = r_t[5] + discount_t[5] * v_t[6] - v_t[5]  # 1 + 0.9*0 - 1 = 0
        expected_t5 = delta_t5 + discount_t[5] * 1.0 * expected_t6  # 0 + 0.9*0 = 0

        # For t=4 (truncation point), the advantage is just the TD error (no bootstrapping)
        expected_t4 = r_t[4] + discount_t[4] * v_t[5] - v_t[4]  # 0 + 0.9*1 - 0 = 0.9

        # For t=3, advantage includes t=3's TD error and discounted advantage from t=4
        delta_t3 = r_t[3] + discount_t[3] * v_t[4] - v_t[3]  # 0 + 0.9*0 - 1 = -1
        expected_t3 = (
            delta_t3 + discount_t[3] * 1.0 * expected_t4
        )  # -1 + 0.9*0.9 = -0.19

        # For t=2 (truncation point), the advantage is just the TD error (no bootstrapping)
        expected_t2 = r_t[2] + discount_t[2] * v_t[3] - v_t[2]  # 0 + 0.9*1 - 2 = -1.1

        # For t=1, advantage includes t=1's TD error and discounted advantage from t=2
        delta_t1 = r_t[1] + discount_t[1] * v_t[2] - v_t[1]  # 1 + 0.9*2 - 1 = 1.8
        expected_t1 = (
            delta_t1 + discount_t[1] * 1.0 * expected_t2
        )  # 1.8 + 0.9*(-1.1) = 0.81

        # For t=0, advantage includes t=0's TD error and discounted advantage from t=1
        delta_t0 = r_t[0] + discount_t[0] * v_t[1] - v_t[0]  # 0 + 0.9*1 - 0 = 0.9
        expected_t0 = (
            delta_t0 + discount_t[0] * 1.0 * expected_t1
        )  # 0.9 + 0.9*0.81 = 1.629

        expected_advantages = jnp.array(
            [
                expected_t0,
                expected_t1,
                expected_t2,
                expected_t3,
                expected_t4,
                expected_t5,
                expected_t6,
            ]
        )

        # Expected targets are advantages + values
        expected_targets = expected_advantages + v_t[:-1]

        np.testing.assert_allclose(advantages, expected_advantages, atol=1e-3)
        np.testing.assert_allclose(targets, expected_targets, atol=1e-3)

    @chex.all_variants()
    def test_truncation_vs_termination(self):
        """Tests that truncation and termination (zero discount) are handled differently."""
        # Create test data with identical structure but one using truncation and one using termination
        r_t = jnp.array([0.0, 0.0, 0.0, 0.0])
        v_t = jnp.array(
            [1.0, 1.0, 1.0, 1.0, 10.0]
        )  # High terminal value to make difference obvious

        # Case 1: Using truncation at t=2
        discount_t_trunc = jnp.array([0.9, 0.9, 0.9, 0.9])
        truncation_t = jnp.array([0.0, 0.0, 1.0, 0.0])

        # Case 2: Using termination at t=2 (via zero discount)
        discount_t_term = jnp.array([0.9, 0.9, 0.0, 0.9])

        r_t, discount_t_trunc, v_t, truncation_t, discount_t_term = (
            jax.tree_util.tree_map(
                lambda x: jnp.expand_dims(x, axis=0),
                (r_t, discount_t_trunc, v_t, truncation_t, discount_t_term),
            )
        )

        # Compute advantages for both cases
        advantage_fn_variant = self.variant(batch_truncated_generalized_advantage_estimation)

        # Truncation case (should bootstrap value at t=3 but not propagate beyond t=2)
        trunc_advantages, trunc_targets = advantage_fn_variant(
            r_t, discount_t_trunc, 1.0, v_t, truncation_t=truncation_t
        )

        # Termination case (should not bootstrap value at t=3)
        term_advantages, term_targets = advantage_fn_variant(
            r_t, discount_t_term, 1.0, v_t, None
        )

        trunc_advantages, trunc_targets, term_advantages, term_targets = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0),
            (trunc_advantages, trunc_targets, term_advantages, term_targets),
        )
        r_t, discount_t_trunc, v_t, truncation_t, discount_t_term = (
            jax.tree_util.tree_map(
                lambda x: jnp.squeeze(x, axis=0),
                (r_t, discount_t_trunc, v_t, truncation_t, discount_t_term),
            )
        )

        # The truncation case should incorporate the high terminal value at t=2
        # while the termination case should not
        delta_t2_trunc = (
            r_t[2] + discount_t_trunc[2] * v_t[3] - v_t[2]
        )  # 0 + 0.9*1 - 1 = -0.1
        delta_t2_term = (
            r_t[2] + discount_t_term[2] * v_t[3] - v_t[2]
        )  # 0 + 0.0*1 - 1 = -1.0

        # Check that t=2 advantages differ between truncation and termination
        np.testing.assert_allclose(trunc_advantages[2], delta_t2_trunc, atol=1e-3)
        np.testing.assert_allclose(term_advantages[2], delta_t2_term, atol=1e-3)

        # Check targets as well
        np.testing.assert_allclose(trunc_targets[2], v_t[2] + delta_t2_trunc, atol=1e-3)
        np.testing.assert_allclose(term_targets[2], v_t[2] + delta_t2_term, atol=1e-3)

        # Values at t=0 and t=1 should bootstrap differently too
        self.assertFalse(
            np.allclose(trunc_advantages[0:2], term_advantages[0:2], atol=1e-3)
        )

        # Values at t=3 should be identical (after both truncation and termination points)
        np.testing.assert_allclose(trunc_advantages[3], term_advantages[3], atol=1e-3)

    @chex.all_variants()
    @parameterized.named_parameters(
        ("lambda_1", 1.0), ("lambda_0.7", 0.7), ("lambda_0.4", 0.4)
    )
    def test_gae_as_special_case_of_importance_corrected_td_errors(self, lambda_):
        """Tests GAE yields same output as importance corrected td errors with dummy ratios."""
        # Test batched GAE
        batched_gae_fn_variant = self.variant(
            batch_truncated_generalized_advantage_estimation
        )
        gae_advantages, gae_targets = batched_gae_fn_variant(
            self.r_t, self.discount_t, lambda_, self.v_t, None
        )

        # Test batched importance corrected td errors
        batched_ictd_errors_fn_variant = self.variant(
            jax.vmap(importance_corrected_td_errors)
        )
        ictd_errors_result = batched_ictd_errors_fn_variant(
            self.r_t,
            self.discount_t,
            self.dummy_rho_tm1,
            jnp.ones_like(self.discount_t) * lambda_,
            self.v_t,
            None,
        )

        np.testing.assert_allclose(gae_advantages, ictd_errors_result, atol=1e-3)

    @chex.all_variants()
    def test_gae_with_truncation_as_special_case_of_importance_corrected_td_errors(
        self,
    ):
        """Tests truncated GAE with truncation.

        Tests that truncated GAE with truncation yields same output as importance
        corrected td errors with dummy ratios and truncation.
        """
        # Create a version of discount_t without the zero at the truncation point
        modified_discount_t = jnp.array(
            [[0.99, 0.99, 0.99, 0.99, 0.99], [0.9, 0.9, 0.9, 0.9, 0.9]]
        )

        # Test with GAE truncation
        batched_gae_fn_variant = self.variant(
            batch_truncated_generalized_advantage_estimation
        )
        gae_advantages, _ = batched_gae_fn_variant(
            self.r_t, modified_discount_t, 1.0, self.v_t, truncation_t=self.truncation_t
        )

        # Test with importance corrected td errors truncation
        batched_ictd_errors_fn_variant = self.variant(
            jax.vmap(importance_corrected_td_errors)
        )
        ictd_errors_result = batched_ictd_errors_fn_variant(
            self.r_t,
            modified_discount_t,
            self.dummy_rho_tm1,
            jnp.ones_like(self.discount_t),
            self.v_t,
            truncation_t=self.truncation_t,
        )

        np.testing.assert_allclose(gae_advantages, ictd_errors_result, atol=1e-3)

    @chex.all_variants()
    def test_truncation_and_termination_combined(self):
        """Tests GAE with a mix of truncation and termination conditions."""
        # Create a batch with both truncation and termination in different episodes
        # A batch with 3 episodes:
        # - First episode: normal with terminal state
        # - Second episode: truncated
        # - Third episode: normal with terminal state
        r_t = jnp.array(
            [
                [1.0, 0.0, 0.0],  # Episode 1 rewards
                [0.0, 0.5, 0.0],  # Episode 2 rewards
                [0.0, 0.0, 2.0],  # Episode 3 rewards
            ]
        )

        v_t = jnp.array(
            [
                [1.0, 1.0, 0.0, 0.0],  # Episode 1 values
                [1.0, 1.0, 1.0, 0.0],  # Episode 2 values
                [1.0, 1.0, 0.5, 0.0],  # Episode 3 values
            ]
        )

        # Discount for terminal states is 0
        discount_t = jnp.array(
            [
                [0.9, 0.9, 0.0],  # Episode 1 - terminal at t=2
                [0.9, 0.9, 0.9],  # Episode 2 - no termination
                [0.9, 0.9, 0.0],  # Episode 3 - terminal at t=2
            ]
        )

        # Truncation indicator
        truncation_t = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Episode 1 - no truncation
                [0.0, 0.0, 1.0],  # Episode 2 - truncated at t=2
                [0.0, 0.0, 0.0],  # Episode 3 - no truncation
            ]
        )

        # Calculate advantages using our function
        batched_advantage_fn_variant = self.variant(
            batch_truncated_generalized_advantage_estimation
        )

        advantages, targets = batched_advantage_fn_variant(
            r_t, discount_t, 1.0, v_t, truncation_t=truncation_t
        )

        # Calculate expected advantages manually for each episode

        # Episode 1 (normal with terminal at t=2)
        # t=2 (terminal)
        delta_t2_ep1 = (
            r_t[0, 2] + discount_t[0, 2] * v_t[0, 3] - v_t[0, 2]
        )  # 0 + 0*0 - 0 = 0
        expected_t2_ep1 = delta_t2_ep1  # 0

        # t=1
        delta_t1_ep1 = (
            r_t[0, 1] + discount_t[0, 1] * v_t[0, 2] - v_t[0, 1]
        )  # 0 + 0.9*0 - 1 = -1
        expected_t1_ep1 = (
            delta_t1_ep1 + discount_t[0, 1] * 1.0 * expected_t2_ep1
        )  # -1 + 0.9*0 = -1

        # t=0
        delta_t0_ep1 = (
            r_t[0, 0] + discount_t[0, 0] * v_t[0, 1] - v_t[0, 0]
        )  # 1 + 0.9*1 - 1 = 0.9
        expected_t0_ep1 = (
            delta_t0_ep1 + discount_t[0, 0] * 1.0 * expected_t1_ep1
        )  # 0.9 + 0.9*(-1) = 0.9 - 0.9 = 0

        # Episode 2 (truncated at t=2)
        # t=2 (truncated)
        delta_t2_ep2 = (
            r_t[1, 2] + discount_t[1, 2] * v_t[1, 3] - v_t[1, 2]
        )  # 0 + 0.9*0 - 1 = -1
        expected_t2_ep2 = (
            delta_t2_ep2  # -1 (since truncated, no bootstrapping beyond this point)
        )

        # t=1
        delta_t1_ep2 = (
            r_t[1, 1] + discount_t[1, 1] * v_t[1, 2] - v_t[1, 1]
        )  # 0.5 + 0.9*1 - 1 = 0.4
        expected_t1_ep2 = (
            delta_t1_ep2 + discount_t[1, 1] * 1.0 * expected_t2_ep2
        )  # 0.4 + 0.9*(-1) = 0.4 - 0.9 = -0.5

        # t=0
        delta_t0_ep2 = (
            r_t[1, 0] + discount_t[1, 0] * v_t[1, 1] - v_t[1, 0]
        )  # 0 + 0.9*1 - 1 = -0.1
        expected_t0_ep2 = (
            delta_t0_ep2 + discount_t[1, 0] * 1.0 * expected_t1_ep2
        )  # -0.1 + 0.9*(-0.5) = -0.1 - 0.45 = -0.55

        # Episode 3 (normal with terminal at t=2)
        # t=2 (terminal)
        delta_t2_ep3 = (
            r_t[2, 2] + discount_t[2, 2] * v_t[2, 3] - v_t[2, 2]
        )  # 2 + 0*0 - 0.5 = 1.5
        expected_t2_ep3 = delta_t2_ep3  # 1.5

        # t=1
        delta_t1_ep3 = (
            r_t[2, 1] + discount_t[2, 1] * v_t[2, 2] - v_t[2, 1]
        )  # 0 + 0.9*0.5 - 1 = -0.55
        expected_t1_ep3 = (
            delta_t1_ep3 + discount_t[2, 1] * 1.0 * expected_t2_ep3
        )  # -0.55 + 0.9*1.5 = -0.55 + 1.35 = 0.8

        # t=0
        delta_t0_ep3 = (
            r_t[2, 0] + discount_t[2, 0] * v_t[2, 1] - v_t[2, 0]
        )  # 0 + 0.9*1 - 1 = -0.1
        expected_t0_ep3 = (
            delta_t0_ep3 + discount_t[2, 0] * 1.0 * expected_t1_ep3
        )  # -0.1 + 0.9*0.8 = -0.1 + 0.72 = 0.62

        # Combine all expected advantages
        expected_advantages = jnp.array(
            [
                [expected_t0_ep1, expected_t1_ep1, expected_t2_ep1],
                [expected_t0_ep2, expected_t1_ep2, expected_t2_ep2],
                [expected_t0_ep3, expected_t1_ep3, expected_t2_ep3],
            ]
        )

        # Calculate expected targets (values + advantages)
        expected_targets = jnp.array(
            [
                [
                    v_t[0, 0] + expected_t0_ep1,
                    v_t[0, 1] + expected_t1_ep1,
                    v_t[0, 2] + expected_t2_ep1,
                ],
                [
                    v_t[1, 0] + expected_t0_ep2,
                    v_t[1, 1] + expected_t1_ep2,
                    v_t[1, 2] + expected_t2_ep2,
                ],
                [
                    v_t[2, 0] + expected_t0_ep3,
                    v_t[2, 1] + expected_t1_ep3,
                    v_t[2, 2] + expected_t2_ep3,
                ],
            ]
        )

        # Check if actual advantages match expected advantages
        np.testing.assert_allclose(advantages, expected_advantages, atol=1e-3)
        # Check if actual targets match expected targets
        np.testing.assert_allclose(targets, expected_targets, atol=1e-3)


if __name__ == "__main__":
    jax.config.update("jax_numpy_rank_promotion", "raise")
    absltest.main()
