"""
MuJoCo Playground wrapper for stoix compatibility.

This wrapper adapts MuJoCo Playground environments to work with the stoix
environment interface, following the same pattern as the Brax wrapper.
"""

from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import StepType, TimeStep, restart
from mujoco_playground import MjxEnv

from stoix.base_types import Observation


@struct.dataclass
class MuJoCoPlaygroundState:
    """State container for MuJoCo Playground environments."""
    mjx_state: Any
    obs: chex.Array
    reward: chex.Numeric
    done: chex.Numeric
    key: chex.PRNGKey
    step_count: chex.Array
    metrics: Dict[str, jnp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class MuJoCoPlaygroundWrapper(Environment):
    """Wrapper for MuJoCo Playground environments."""
    
    def __init__(self, env: MjxEnv):
        """Initialize the wrapper.
        
        Args:
            env: MuJoCo Playground environment instance.
        """
        self._env = env
        
        if hasattr(self._env, 'observation_size'):
            # Get observation size from environment
            self._obs_size = self._env.observation_size
        else:
            raise ValueError("MuJoCo Playground environment must have 'observation_size' attribute.")
        
        # Get action size from environment
        if hasattr(self._env, 'action_size'):
            self._action_size = self._env.action_size
        else:
            raise ValueError("MuJoCo Playground environment must have 'action_size' attribute.")
    
    def reset(self, key: chex.PRNGKey) -> Tuple[MuJoCoPlaygroundState, TimeStep]:
        """Reset the environment.
        
        Args:
            key: JAX random key for stochastic reset
            
        Returns:
            Tuple of (state, timestep) for initial step
        """
        # Reset the underlying MJX environment
        mjx_state = self._env.reset(key)
        
        # Create wrapped state
        state = MuJoCoPlaygroundState(
            mjx_state=mjx_state,
            obs=mjx_state.obs,
            reward=mjx_state.reward,
            done=mjx_state.done,
            key=key,
            step_count=jnp.zeros((), dtype=jnp.int32),
            metrics={**mjx_state.metrics},
            info={**mjx_state.info},
        )
        
        # Create initial observation
        agent_view = mjx_state.obs.astype(jnp.float32)
        legal_action_mask = jnp.ones((self._action_size,), dtype=jnp.float32)
        
        timestep = restart(
            observation=Observation(
                agent_view,
                legal_action_mask,
                state.step_count,
            ),
            extras={},
        )
        
        return state, timestep
    
    def step(self, state: MuJoCoPlaygroundState, action: chex.Array) -> Tuple[MuJoCoPlaygroundState, TimeStep]:
        """Step the environment forward.
        
        Args:
            state: Current environment state
            action: Action to take
            
        Returns:
            Tuple of (next_state, timestep)
        """
        # Get previous terminal status
        prev_done = state.done
        
        # Step the MJX environment
        mjx_state = self._env.step(state.mjx_state, action)
        
        # Update step count
        new_step_count = state.step_count + 1
        
        # Extract step information
        obs = mjx_state.obs
        reward = mjx_state.reward
        done = mjx_state.done
        
        # Handle info/metrics dict
        info = mjx_state.info
        metrics = mjx_state.metrics
        
        # Check for truncation (if specified in info)
        truncated = info.get('truncated', jnp.zeros((), dtype=jnp.bool_))
        terminated = jnp.logical_and(done, jnp.logical_not(truncated))
        
        # Calculate discount (0 if terminated, 1 if continuing or truncated)
        discount = jnp.where(terminated, 0.0, 1.0)
        # If previous step was done, discount should be 0
        discount = jnp.where(prev_done, 0.0, discount)
        
        # Determine step type
        step_type = jnp.where(
            done,
            StepType.LAST,
            StepType.MID
        )
        
        # Create new state
        next_state = MuJoCoPlaygroundState(
            mjx_state=mjx_state,
            obs=obs,
            reward=reward,
            done=done,
            key=state.key,
            step_count=new_step_count,
            metrics=metrics,
            info=info,
        )
        
        # Create observation
        agent_view = obs.astype(jnp.float32)
        legal_action_mask = jnp.ones((self._action_size,), dtype=jnp.float32)
        
        observation = Observation(
            agent_view,
            legal_action_mask,
            new_step_count,
        )
        
        # Create timestep
        timestep = TimeStep(
            step_type=step_type,
            reward=reward.astype(jnp.float32),
            discount=discount.astype(jnp.float32),
            observation=observation,
            extras={},
        )
        
        return next_state, timestep
    
    def observation_spec(self) -> specs.Spec:
        """Return the observation specification."""
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=specs.Array(shape=(self._obs_size,), dtype=jnp.float32),
            action_mask=specs.Array(shape=(self._action_size,), dtype=jnp.float32),
            step_count=specs.Array(shape=(), dtype=jnp.int32),
        )
    
    def action_spec(self) -> specs.BoundedArray:
        """Return the action specification."""
        return specs.BoundedArray(
            shape=(self._action_size,),
            dtype=jnp.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )
    
    def reward_spec(self) -> specs.Array:
        """Return the reward specification."""
        return specs.Array(shape=(), dtype=jnp.float32, name="reward")
    
    def discount_spec(self) -> specs.BoundedArray:
        """Return the discount specification."""
        return specs.BoundedArray(
            shape=(), 
            dtype=jnp.float32, 
            minimum=0.0, 
            maximum=1.0, 
            name="discount"
        )