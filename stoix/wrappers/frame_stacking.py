from typing import TYPE_CHECKING, Sequence, Tuple

import chex
import jax.numpy as jnp
import jumanji.specs as specs
from jumanji.env import Environment, State
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

# This wrapper is experimental and has not been tested


@dataclass
class StackState:
    stacked_frames: chex.Array


class FrameStacker:
    def __init__(self, num_frames: int, frame_shape: Sequence[int], flatten: bool = False):
        self._num_frames = num_frames
        self._flatten = flatten
        self._frame_shape = frame_shape

    def reset(self) -> StackState:
        stacked_frames = jnp.zeros(
            (
                *self._frame_shape,
                self._num_frames,
            )
        )
        return StackState(stacked_frames=stacked_frames)

    def step(self, stack_state: StackState, frame: chex.Array) -> StackState:
        # Shift the frames and add the new frame to the end
        stacked_frames = jnp.roll(stack_state.stacked_frames, shift=-1, axis=-1)
        stacked_frames = stacked_frames.at[..., -1].set(frame)
        return StackState(stacked_frames=stacked_frames)


@dataclass
class FrameStackEnvState:
    env_state: State
    stack_state: StackState


class FrameStackingWrapper(Wrapper):
    """Wrapper that stacks observations along a new final axis."""

    def __init__(self, env: Environment, num_frames: int = 4, flatten: bool = True):
        """Initializes a new FrameStackingWrapper.

        Args:
            env: environment to wrap.
            num_frames: Number frames to stack.
            flatten: Whether to flatten the channel and stacking dimensions together.
                e.g. (H, W, C, num_frames) -> (H, W, C * num_frames)
        """
        super().__init__(env)
        original_spec = self._env.observation_spec()
        # We only stack the agent view
        self._stacker = FrameStacker(
            num_frames=num_frames, flatten=flatten, frame_shape=original_spec.agent_view.shape
        )
        self._num_frames = num_frames
        self._flatten = flatten

    def update_spec(self, spec: specs.Spec) -> specs.Spec:
        if not self._flatten:
            new_shape = spec.shape + (self._num_frames,)
        else:
            new_shape = spec.shape[:-1] + (self._num_frames * spec.shape[-1],)
        if type(spec) is specs.Array:
            return spec.replace(shape=new_shape)
        elif type(spec) is specs.BoundedArray:
            if spec.minimum.shape != ():
                new_minimum = jnp.repeat(spec.minimum, self._num_frames, axis=-1).reshape(new_shape)
                new_maximum = jnp.repeat(spec.maximum, self._num_frames, axis=-1).reshape(new_shape)
            else:
                new_minimum = spec.minimum
                new_maximum = spec.maximum
            return spec.replace(shape=new_shape, minimum=new_minimum, maximum=new_maximum)
        else:
            raise ValueError(f"Unsupported spec type {type(spec)}")

    def stacked_frames_to_view(self, stacked_frames: chex.Array) -> chex.Array:
        if not self._flatten:
            return stacked_frames
        else:
            new_shape = stacked_frames.shape[:-2] + (-1,)
            return stacked_frames.reshape(*new_shape)

    def _process_timestep(self, stack_state: StackState, timestep: TimeStep) -> TimeStep:
        observation = timestep.observation
        agent_view = observation.agent_view
        new_stack_state = self._stacker.step(stack_state, agent_view)
        observation = observation._replace(
            agent_view=self.stacked_frames_to_view(new_stack_state.stacked_frames)
        )
        return new_stack_state, timestep.replace(observation=observation)

    def reset(self, key: chex.PRNGKey) -> Tuple[FrameStackEnvState, TimeStep]:
        stack_state = self._stacker.reset()
        env_state, timestep = self._env.reset(key)
        new_stack_state, timestep = self._process_timestep(stack_state, timestep)
        stacked_env_state = FrameStackEnvState(env_state=env_state, stack_state=new_stack_state)
        return stacked_env_state, timestep

    def step(
        self, state: FrameStackEnvState, action: chex.Array
    ) -> Tuple[FrameStackEnvState, TimeStep]:
        env_state, timestep = self._env.step(state.env_state, action)
        new_stack_state, timestep = self._process_timestep(state.stack_state, timestep)
        return FrameStackEnvState(env_state=env_state, stack_state=new_stack_state), timestep

    def observation_spec(self) -> specs.Spec:
        spec = self._env.observation_spec()
        spec = spec.replace(agent_view=self.update_spec(spec.agent_view))
        return spec
