import jax.numpy as jnp
import jax
from functools import partial


@partial(jax.jit, static_argnames=("capacity", "obs_shape"))
def initialize_memory(capacity, obs_shape):
    # Pre-allocate JAX arrays for efficiency
    obss = jnp.zeros((capacity, *obs_shape), dtype=jnp.uint8)
    actions = jnp.zeros(capacity, dtype=jnp.uint8)
    rewards = jnp.zeros(capacity, dtype=jnp.float32)
    next_obss = jnp.zeros((capacity, *obs_shape), dtype=jnp.uint8)
    terminals = jnp.zeros(capacity, dtype=jnp.bool_)

    return obss, actions, rewards, next_obss, terminals


@jax.jit
def convert_dtypes(
        obs: jnp.array,
        action: jnp.array,
        reward: jnp.array,
        next_obs: jnp.array,
        terminal: jnp.array):
    # convert dtypes
    obs_arr = jnp.array(obs, dtype=jnp.uint8)
    action_arr = jnp.array(action, dtype=jnp.uint8)
    reward_arr = jnp.array(reward, jnp.float32)
    next_obs_arr = jnp.array(next_obs, dtype=jnp.uint8)
    terminal_arr = jnp.array(terminal, dtype=jnp.bool_)

    return obs_arr, action_arr, reward_arr, next_obs_arr, terminal_arr
    

@partial(jax.jit, donate_argnames=(
    "obss", "actions", "rewards", "next_obss", "terminals", 
    "pointer"))
def update_memory(
    obss: jnp.array,
    actions: jnp.array,
    rewards: jnp.array,
    next_obss: jnp.array,
    terminals: jnp.array,
    pointer: jnp.array,
    obs: jnp.array,
    action: jnp.array,
    reward: jnp.array,
    next_obs: jnp.array,
    terminal: jnp.array,
):
    # update entries at pointer
    obss.at[pointer].set(obs)
    actions.at[pointer].set(action)
    rewards.at[pointer].set(jnp.sign(reward))
    next_obss.at[pointer].set(next_obs)
    terminals.at[pointer].set(terminal)
    return obss, actions, rewards, next_obss, terminals

    
@partial(jax.jit, donate_argnames=("obss", "actions", "rewards", "next_obss", "terminals", 
    "pointer", "curr_size"))
def batch_update_memory(
    obss: jnp.array,
    actions: jnp.array,
    rewards: jnp.array,
    next_obss: jnp.array,
    terminals: jnp.array,
    pointer: jnp.array,
    capacity: jnp.array,
    curr_size: jnp.array,
    obs: jnp.array,
    action: jnp.array,
    reward: jnp.array,
    next_obs: jnp.array,
    terminal: jnp.array,
):
    batch_size = len(obs)
    indices = (pointer + jnp.arange(batch_size)) % capacity

    updated_obss = obss.at[indices].set(obs)
    updated_actions = actions.at[indices].set(action)
    updated_rewards = rewards.at[indices].set(jnp.sign(jnp.array(reward)))
    updated_next_obss = next_obss.at[indices].set(next_obs)
    updated_terminals = terminals.at[indices].set(terminal)

    # update pointer and size
    pointer = (pointer + batch_size) % capacity
    new_size = jnp.minimum(curr_size + batch_size, capacity)

    return updated_obss, updated_actions, updated_rewards, updated_next_obss, updated_terminals, pointer, new_size

#@partial(jax.jit, static_argnames=["batch_size"])
def retrieve_experience(
    obss: jnp.array, 
    actions: jnp.array,
    rewards: jnp.array,
    next_obss: jnp.array,
    terminals: jnp.array,
    batch_size: int, 
    size: int,
    key: jax.random.PRNGKey
):
    # split random key
    new_key, subkey = jax.random.split(key)

    # generate random indices
    indices = jax.random.randint(
        key=subkey, shape=(batch_size,), minval=0, maxval=size
    )

    return obss[indices], actions[indices], rewards[indices], next_obss[indices], terminals[indices].astype(jnp.float32), new_key
