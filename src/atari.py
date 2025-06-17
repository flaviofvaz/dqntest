import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
from gymnasium.core import ObservationWrapper, Wrapper
from typing import Tuple
from gymnasium.spaces import Box
import numpy as np
from collections import deque
import jax.numpy as jnp
import jax


class AtariEnvironment:
    def __init__(self, environment_name: str, grayscale: bool=True):
        gym.register_envs(ale_py)
        env = gym.make(f'{environment_name}', max_episode_steps=108_000)
        env = AtariPreprocessing(env)
        env = StackFrames(env, num_stack=4)
        self._env = JaxConversionWrapper(env)
        self._lives = None
    
    def step(self, action: int) -> Tuple:
        life_loss = False
        new_observation, reward, terminated, truncated, info = self._env.step(action)
        if self._lives is not None and "lives" in info:
            if info["lives"] < self._lives:
                self._lives = info["lives"]
                life_loss = True 
        return new_observation, reward, terminated, truncated, life_loss
    
    def reset(self, **kwargs) -> Tuple:
        obs, info = self._env.reset(**kwargs)
        if "lives" in info:
            self._lives = info["lives"]
        return obs, info
    
    def close(self) -> None:
        self._env.close()

    def get_action_space(self) -> int:
        return self._env.action_space.n
    
    def sample_action(self) -> int:
        return jnp.array(self._env.action_space.sample(), jnp.uint8)
    

class StackFrames(ObservationWrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        self.original_obs_shape = self.observation_space.shape

       # Modify the observation space
        if len(self.original_obs_shape) == 2: # Handles (H, W) case like (84, 84)
            # New space shape will be (H, W, num_stack) e.g., (84, 84, 4)
            low = np.expand_dims(self.observation_space.low, axis=-1)
            high = np.expand_dims(self.observation_space.high, axis=-1)
            low = np.repeat(low, num_stack, axis=-1)
            high = np.repeat(high, num_stack, axis=-1)
            self.dtype = self.observation_space.dtype
        elif len(self.original_obs_shape) == 3: # Handles (H, W, C)
            low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
            high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
            self.dtype = self.observation_space.dtype
        else:
            raise ValueError(
                f"Observation space shape {self.original_obs_shape} not supported. "
                "Expected 2D (HxW) or 3D (HxWxC)."
            )

        self.observation_space = Box(
            low=low, high=high, dtype=self.dtype
        )

    def _process_frame(self, observation):
        # If original observation was 2D (e.g., (84,84)), expand it to (H, W, 1) e.g. (84,84,1)
        if len(self.original_obs_shape) == 2:
            return np.expand_dims(observation, axis=-1)
        return observation # If already 3D (H,W,C), return as is

    def observation(self, observation):
        # Process the incoming frame (e.g., ensure it's (H,W,1) if original was (H,W))
        processed_frame = self._process_frame(observation)
        self.frames.append(processed_frame)

        # Padding logic if deque isn't full yet (typically at the start)
        if len(self.frames) < self.num_stack:
            frames_to_repeat = list(self.frames)
            while len(self.frames) < self.num_stack:
                for frame_to_add in frames_to_repeat:
                    if len(self.frames) < self.num_stack:
                        self.frames.appendleft(frame_to_add)
                    else:
                        break
        
        # Concatenate frames in the deque along the last axis.
        # If frames are (H,W,1), this results in (H,W,num_stack)
        return np.concatenate(list(self.frames), axis=-1)


    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # Process the initial frame and fill the deque
        processed_frame = self._process_frame(observation)
        self.frames.clear() # Clear frames at reset
        for _ in range(self.num_stack):
            self.frames.append(processed_frame)
        
        # Return the first stacked observation. Note: self.observation() is called
        # with the raw observation, and it will process it internally.
        return self.observation(observation), info
    

class JaxConversionWrapper(Wrapper):
    """
    A Gymnasium wrapper that converts observations, rewards, and termination
    flags to JAX arrays with specified dtypes.

    - Observations are converted to jnp.array with dtype jnp.uint8.
    - Rewards are converted to jnp.array with dtype jnp.float32.
    - Terminated and Truncated flags are converted to jnp.array with dtype jnp.bool_.

    The observation space (if it's a gymnasium.spaces.Box) will have its
    dtype attribute updated to np.uint8. Users should be aware of potential
    implications if the original observation data range doesn't naturally fit
    uint8 (e.g., float values between 0.0-1.0 would be cast to 0 if not scaled).
    """
    def __init__(self, env):
        super().__init__(env)

        # Modify the observation space to reflect the new dtype.
        # Note: Gymnasium spaces are defined using NumPy dtypes.
        current_obs_space = self.env.observation_space
        if isinstance(current_obs_space, gym.spaces.Box):
            # We update the dtype of the space. The original low/high bounds
            # are preserved. The user should be aware that if the original
            # dtype was float, direct casting to uint8 might alter the
            # interpretation of these bounds if data is not scaled appropriately
            # (e.g. 0.0-1.0 float to 0-255 uint8). This wrapper performs a
            # direct cast as per the request for jnp.uint8 observations.
            self.observation_space = gym.spaces.Box(
                low=current_obs_space.low,
                high=current_obs_space.high,
                shape=current_obs_space.shape,
                dtype=np.uint8  # Corresponds to jnp.uint8
            )
        # If the original observation space is not a Box (e.g., Discrete),
        # its definition is not changed here, but the returned observation
        # will still be a jnp.array of the discrete value, cast to jnp.uint8.

        # Action space and reward range are not explicitly changed in definition,
        # but rewards will be returned as jnp.float32.

    def reset(self, **kwargs):
        """
        Resets the environment and converts the observation to a jnp.array (uint8).
        """
        obs, info = self.env.reset(**kwargs)
        jax_obs = jnp.array(obs, dtype=jnp.uint8)
        # Optionally convert NumPy arrays in info to JAX arrays
        # jax_info = self._convert_info_to_jax(info)
        return jax_obs, info # Or jax_info if conversion is applied

    def step(self, action):
        """
        Steps through the environment, converts the observation (uint8),
        reward (float32), terminated (bool_), and truncated (bool_)
        to jnp.arrays.
        Converts action from JAX array to NumPy array if necessary for the
        underlying environment.
        """
        # If the action is a JAX array, convert it to NumPy for the underlying environment
        if isinstance(action, jax.Array):
            action = np.array(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        jax_obs = jnp.array(obs, dtype=jnp.uint8)
        jax_reward = jnp.array(reward, dtype=jnp.float32)
        jax_terminated = jnp.array(terminated, dtype=jnp.bool_)
        jax_truncated = jnp.array(truncated, dtype=jnp.bool_) # Handle truncated as well

        # Optionally convert NumPy arrays in info to JAX arrays
        # jax_info = self._convert_info_to_jax(info)
        return jax_obs, jax_reward, jax_terminated, jax_truncated, info # Or jax_info