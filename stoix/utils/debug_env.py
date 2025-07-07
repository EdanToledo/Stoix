from typing import TYPE_CHECKING, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from stoa.env_types import Action, EnvParams, StepType, TimeStep
from stoa.environment import Environment
from stoa.spaces import ArraySpace, DiscreteSpace, Space

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class GameState:
    """Minimal state for debug environments."""

    step_count: Array
    state: Array
    key: PRNGKey


class IdentityGame(Environment):
    """Minimal debug environment where the agent must predict the current state.

    At each step, shows a random number from 0 to num_actions-1.
    Agent gets reward 1.0 if action matches the number, 0.0 otherwise.
    Episodes last 50 steps.
    """

    def __init__(self, num_actions: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.max_steps = 50

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[GameState, TimeStep]:
        state_key, next_key = jax.random.split(rng_key)
        state_val = jax.random.randint(state_key, shape=(), minval=0, maxval=self.num_actions)

        state = GameState(state=state_val, key=next_key, step_count=jnp.array(0, dtype=jnp.int32))

        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=state_val.astype(jnp.float32),
            extras={},
        )

        return state, timestep

    def step(
        self,
        state: GameState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[GameState, TimeStep]:
        # Reward for correct prediction
        reward = jnp.where(action == state.state, 1.0, 0.0)

        # Generate next random state
        state_key, next_key = jax.random.split(state.key)
        next_state_val = jax.random.randint(state_key, shape=(), minval=0, maxval=self.num_actions)

        new_step_count = state.step_count + 1

        new_state = GameState(state=next_state_val, key=next_key, step_count=new_step_count)

        # Check if episode is done
        done = new_step_count >= self.max_steps
        step_type = jnp.where(done, StepType.TERMINATED, StepType.MID)
        discount = jnp.where(done, 0.0, 1.0)

        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            observation=next_state_val.astype(jnp.float32),
            extras={},
        )

        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.float32, name="observation")

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return DiscreteSpace(num_values=self.num_actions, dtype=jnp.int32, name="action")

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.int32, name="state")


class SequenceGame(Environment):
    """Minimal sequence prediction debug environment.

    Shows numbers in sequence: 0, 1, 2, ..., num_actions-1, 0, 1, ...
    Agent must predict the current number to get reward.
    Episodes last 50 steps.
    """

    def __init__(self, num_actions: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.max_steps = 50

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[GameState, TimeStep]:
        state_key, next_key = jax.random.split(rng_key)
        state_val = jax.random.randint(state_key, shape=(), minval=0, maxval=self.num_actions)

        state = GameState(state=state_val, key=next_key, step_count=jnp.array(0, dtype=jnp.int32))

        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=state_val.astype(jnp.float32),
            extras={},
        )

        return state, timestep

    def step(
        self,
        state: GameState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[GameState, TimeStep]:
        # Reward for correct prediction
        reward = jnp.where(action == state.state, 1.0, 0.0)

        # Next state is sequential (current + 1) mod num_actions
        next_state_val = (state.state + 1) % self.num_actions

        new_step_count = state.step_count + 1

        # Keep the same key (no randomness needed for deterministic sequence)
        new_state = GameState(state=next_state_val, key=state.key, step_count=new_step_count)

        # Check if episode is done
        done = new_step_count >= self.max_steps
        step_type = jnp.where(done, StepType.TERMINATED, StepType.MID)
        discount = jnp.where(done, 0.0, 1.0)

        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            observation=next_state_val.astype(jnp.float32),
            extras={},
        )

        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.float32, name="observation")

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return DiscreteSpace(num_values=self.num_actions, dtype=jnp.int32, name="action")

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.int32, name="state")


class DelayedRewardGame(Environment):
    """Tests if algorithm can handle delayed rewards.

    Agent chooses action 0 or 1. Only action 1 gives reward, but delayed by 5 steps.
    Buggy algorithms that don't handle credit assignment will fail.
    """

    def __init__(self, delay_steps: int = 5):
        super().__init__()
        self.delay_steps = delay_steps
        self.max_steps = 20

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[GameState, TimeStep]:
        state = GameState(
            state=jnp.array(0, dtype=jnp.int32),  # Steps since last action 1
            key=rng_key,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=jnp.array(0.0, dtype=jnp.float32),  # Always 0, no info needed
            extras={},
        )

        return state, timestep

    def step(
        self,
        state: GameState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[GameState, TimeStep]:
        # Give reward if exactly delay_steps ago we took action 1
        reward = jnp.where(state.state == self.delay_steps, 1.0, 0.0)

        # Update counter: reset to 1 if action 1, increment if action 0, cap at delay+1
        new_counter = jnp.where(
            action == 1,
            1,  # Reset to 1 if we took action 1
            jnp.minimum(state.state + 1, self.delay_steps + 1),  # Increment, cap
        )

        new_step_count = state.step_count + 1

        new_state = GameState(state=new_counter, key=state.key, step_count=new_step_count)

        done = new_step_count >= self.max_steps
        step_type = jnp.where(done, StepType.TERMINATED, StepType.MID)
        discount = jnp.where(done, 0.0, 1.0)

        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            observation=jnp.array(0.0, dtype=jnp.float32),
            extras={},
        )

        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.float32, name="observation")

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return DiscreteSpace(num_values=2, dtype=jnp.int32, name="action")

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.int32, name="state")


class DiscountSensitiveGame(Environment):
    """Tests if algorithm properly uses discount factor.

    Two actions: 0 gives +1 now, 1 gives +10 in 3 steps then episode ends.
    With high discount (0.99), action 1 is better. With low discount (0.1), action 0 is better.
    """

    def __init__(self, big_reward_delay: int = 3):
        super().__init__()
        self.big_reward_delay = big_reward_delay
        self.max_steps = 10

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[GameState, TimeStep]:
        state = GameState(
            state=jnp.array(-1, dtype=jnp.int32),  # -1 = no big reward coming, >=0 = countdown
            key=rng_key,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=jnp.array(0.0, dtype=jnp.float32),
            extras={},
        )

        return state, timestep

    def step(
        self,
        state: GameState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[GameState, TimeStep]:
        # Immediate reward for action 0
        immediate_reward = jnp.where(action == 0, 1.0, 0.0)

        # Big reward if countdown reached 0
        big_reward = jnp.where(state.state == 0, 10.0, 0.0)
        total_reward = immediate_reward + big_reward

        # Update countdown: start if action 1 and not already counting, decrement if counting
        new_countdown = jnp.where(
            jnp.logical_and(action == 1, state.state == -1),
            self.big_reward_delay,  # Start countdown
            jnp.where(state.state >= 0, state.state - 1, -1),  # Continue countdown  # Stay at -1
        )

        new_step_count = state.step_count + 1

        new_state = GameState(state=new_countdown, key=state.key, step_count=new_step_count)

        # Episode ends after big reward or max steps
        done = jnp.logical_or(state.state == 0, new_step_count >= self.max_steps)
        step_type = jnp.where(done, StepType.TERMINATED, StepType.MID)
        discount = jnp.where(done, 0.0, 1.0)

        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(total_reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            observation=jnp.array(0.0, dtype=jnp.float32),
            extras={},
        )

        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.float32, name="observation")

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return DiscreteSpace(num_values=2, dtype=jnp.int32, name="action")

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.int32, name="state")


class ExplorationGame(Environment):
    """Tests if algorithm explores properly.

    Action 0 always gives +0.1. Action 1 gives +1.0 but only 10% of the time.
    Expected values: action 0 = 0.1, action 1 = 0.1, so need exploration to find action 1.
    """

    def __init__(self, good_action_prob: float = 0.1):
        super().__init__()
        self.good_action_prob = good_action_prob
        self.max_steps = 100

    def reset(
        self, rng_key: PRNGKey, env_params: Optional[EnvParams] = None
    ) -> Tuple[GameState, TimeStep]:
        state = GameState(
            state=jnp.array(0, dtype=jnp.int32),  # Unused
            key=rng_key,
            step_count=jnp.array(0, dtype=jnp.int32),
        )

        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=jnp.array(0.0, dtype=jnp.float32),
            discount=jnp.array(1.0, dtype=jnp.float32),
            observation=jnp.array(0.0, dtype=jnp.float32),
            extras={},
        )

        return state, timestep

    def step(
        self,
        state: GameState,
        action: Action,
        env_params: Optional[EnvParams] = None,
    ) -> Tuple[GameState, TimeStep]:
        step_key, next_key = jax.random.split(state.key)

        # Action 0: always small reward
        # Action 1: big reward with small probability
        reward = jnp.where(
            action == 0,
            0.1,  # Guaranteed small reward
            jnp.where(
                jax.random.uniform(step_key) < self.good_action_prob,
                1.0,  # Rare big reward
                0.0,  # Usually nothing
            ),
        )

        new_step_count = state.step_count + 1

        new_state = GameState(state=state.state, key=next_key, step_count=new_step_count)

        done = new_step_count >= self.max_steps
        step_type = jnp.where(done, StepType.TERMINATED, StepType.MID)
        discount = jnp.where(done, 0.0, 1.0)

        timestep = TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.asarray(discount, dtype=jnp.float32),
            observation=jnp.array(0.0, dtype=jnp.float32),
            extras={},
        )

        return new_state, timestep

    def observation_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.float32, name="observation")

    def action_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return DiscreteSpace(num_values=2, dtype=jnp.int32, name="action")

    def state_space(self, env_params: Optional[EnvParams] = None) -> Space:
        return ArraySpace(shape=(), dtype=jnp.int32, name="state")


DEBUG_ENVIRONMENTS = {
    "identity": IdentityGame,
    "sequence": SequenceGame,
    "delayed_reward": DelayedRewardGame,
    "discount_sensitive": DiscountSensitiveGame,
    "exploration": ExplorationGame,
}
