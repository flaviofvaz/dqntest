from dqn import DQNAgent, QNetwork
from atari import AtariEnvironment
from flax import nnx
import optax
import jax.numpy as jnp
import jax
import time
import numpy as np
import random as py_random
import rlax
import reverb
import tensorflow as tf

# for cpu development
jax.config.update("jax_platforms", "cpu")

def init_experience_buffer(environment: AtariEnvironment, client: reverb.Client, buffer_name: str, min_capacity: int):
    obs, _ = environment.reset()
    count = 0
    while count < min_capacity:
        action = environment.sample_action()
        next_obs, reward, terminated, truncated, life_loss = environment.step(action)
        terminal_flag = terminated or truncated or life_loss 
        if terminal_flag:
            discount = jnp.array(0.0, dtype=jnp.float32)
        else:
            discount = jnp.array(1.0, dtype=jnp.float32)
        timestep = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "discount": discount,
        }
        client.insert(data=timestep, priorities={buffer_name: 1.0})
        count += 1
        if count % 10000 == 0:
            print(f"{count} transitions added to buffer")
        if terminal_flag:
            obs, _ = environment.reset()
        else:
            obs = next_obs


def init_client(buffer_name: str, max_size: int, min_size: int):
    server = reverb.Server(
        tables=[
            reverb.Table(
                name=buffer_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=max_size,
                rate_limiter=reverb.rate_limiters.MinSize(min_size),
                signature={
                    "obs": tf.TensorSpec([84,84,4], tf.uint8),
                    "action": tf.TensorSpec([None], tf.uint8),
                    "reward": tf.TensorSpec([None], tf.float32),
                    "next_obs": tf.TensorSpec([84,84,4], tf.uint8),
                    "discount": tf.TensorSpec([None], tf.float32)
                }
            ),
        ]
    )
    client = reverb.Client(f"localhost:{server.port}")
    return server, client


_batch_q_learning = jax.vmap(rlax.q_learning)
@nnx.jit
def td_loss(
    model: nnx.Module,
    target_model: nnx.Module,
    obs: jnp.array,
    action: jnp.array,
    reward: jnp.array,
    next_obs: jnp.array,
    discount: jnp.array,
):
    # td estimate
    q_values = model(obs)
    # td target
    target_q_values = jax.lax.stop_gradient(target_model(next_obs))
    # td error
    td_error = _batch_q_learning(q_values, jnp.squeeze(action, axis=-1), jnp.squeeze(reward, axis=-1), 0.99 * jnp.squeeze(discount, axis=-1), target_q_values)
    # clipping gradient and loss
    error_bound = 1.0 / 32
    td_error = rlax.clip_gradient(td_error, -error_bound, error_bound)
    loss = rlax.l2_loss(td_error)
    return jnp.mean(loss)

@nnx.jit
def train_step(
    model: nnx.Module,
    target_model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    obs: jnp.array,
    action: jnp.array,
    reward: jnp.array,
    next_obs: jnp.array,
    discount: jnp.array,
):
    grad_fn = nnx.value_and_grad(td_loss)
    loss, grads = grad_fn(model, target_model, obs, action, reward, next_obs, discount)
    optimizer.update(grads)
    metrics.update(loss=loss)
    

def set_randomness(seed):
    py_random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def train():
    environment_name = "BreakoutNoFrameskip-v4"
    seed = 1
    batch_size = 32
    frames = 10_000_000
    learning_rate = 0.00025
    total_memory_capacity = 100_000
    starting_memory_capacity = 50_000
    observation_space = (84,84,4)
    train_every_n_steps = 4
    update_target_every_n_steps = 10_000
    evaluate_every_n_steps = 250_000
    evaluation_episodes = 100
    evaluation_epsilon = 0.05

    # create environment
    environment = AtariEnvironment(environment_name)

    # initialize main random key
    key = set_randomness(seed)

    # initialize experience memory
    print("Initializing experience memory")
    server, client = init_client("experience_buffer", max_size=total_memory_capacity, min_size=starting_memory_capacity)
    start_time = time.time()
    init_experience_buffer(environment, client, "experience_buffer", starting_memory_capacity)
    print(f"{time.time() - start_time} seconds to initialize memory")

    # splitting random keys
    key, subkey = jax.random.split(key)

    # create network
    q_network = QNetwork(
        input_channels=4, output_dimension=environment.get_action_space(), rngs=nnx.Rngs(params=subkey)
    )

    # create target network
    target_q_network = nnx.clone(q_network)

    # create dqn agent
    dqn_agent = DQNAgent(
        network=q_network, action_space_dim=environment.get_action_space(), gamma=0.99, epsilon=1.0
    )

    # define optimizer
    tx = optax.rmsprop(learning_rate=learning_rate, eps=0.01 / 32**2, centered=True, decay=0.95)
    optimizer = nnx.Optimizer(dqn_agent.network, tx)

    # define metrics
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    # initialize metrics history
    metrics_history = {"train_loss": []}

    print("Starting training")
    start_time = time.time()
    episode = 0
    obs, _ = environment.reset(seed=seed + episode)
    for i in range(frames):
        # get agents action
        action, key = dqn_agent.act(obs, key)

        # perform actions
        next_obs, reward, terminated, truncated, life_loss = environment.step(action)
        terminal_flag = terminated or truncated or life_loss

        # update memory and episode rewards
        if terminal_flag:
            discount = jnp.array(0.0, dtype=jnp.float32)
        else:
            discount = jnp.array(1.0, dtype=jnp.float32)
        timestep = {
            "obs": obs,
            "action": jnp.array(action, dtype=jnp.uint8),
            "reward": reward,
            "next_obs": next_obs,
            "discount": discount,
        }
        client.insert(timestep, priorities={"experience_buffer": 1.0})
        if (i + 1) % train_every_n_steps == 0:
            # sample batch of experiences
            sample = list(client.sample(table="experience_buffer", num_samples=batch_size, emit_timesteps=False, unpack_as_table_signature=True))
            batch = [s.data for s in sample]

            transposed_data = []
            for item in batch:
                transposed_data.append((item['obs'], item['action'], item['reward'], item['next_obs'], item['discount']))

            # Then, use zip(*) to "unzip" it into separate tuples/lists
            obs_batch, action_batch, reward_batch, next_obs_batch, discount_batch = zip(*transposed_data)
            obs_batch = jnp.array(obs_batch)
            action_batch = jnp.array(action_batch)
            reward_batch = jnp.array(reward_batch)
            next_obs_batch = jnp.array(next_obs_batch)
            discount_batch = jnp.array(discount_batch)

            train_step(
                dqn_agent.network,
                target_q_network, 
                optimizer, 
                metrics,
                obs_batch,
                action_batch,
                reward_batch,
                next_obs_batch,
                discount_batch
            )

            # Log training metrics
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)
                metrics.reset()
        
        if (i + 1) % update_target_every_n_steps == 0:
            # update target network
            target_q_network = nnx.clone(q_network)

        if (i + 1) % evaluate_every_n_steps == 0:
            total_rewards = 0.0
            episode += 1
            # evaluate on N episodes
            for _ in range(evaluation_episodes):
                curr_reward, key = run_episode(environment, dqn_agent, key, seed+episode)
                total_rewards += curr_reward
                episode += 1
            total_rewards /= float(evaluation_episodes)
            print(f"Step {i+1} - Evaluation - Average rewards: {total_rewards}")
            obs, _ = environment.reset(seed=seed+episode)
            start_time = time.time()

        if (i + 1) % 10_000 == 0:
            # logging training metrics
            print(
                f"frames seen: {i+1}, "
                f"exploration rate: {dqn_agent.epsilon}, "
                f"loss: {metrics_history['train_loss'][-1]}, "
                f"total time: {time.time() - start_time}"
            )
            start_time = time.time()

        # check if end of game
        if not (terminated or truncated):
            obs = next_obs
        else:
            episode += 1
            obs, _ = environment.reset(seed=seed+episode)


def run_episode(environment, dqn_agent, key, seed):
    done = False
    state, _ = environment.reset(seed=seed)
    total_reward = 0.0
    while not done:
        state = jnp.asarray(state, dtype=jnp.float32)
        agent_action, key = dqn_agent.act(state, key, False)
        new_state, reward, terminated, truncated, _ = environment.step(agent_action)
        total_reward += reward
        state = new_state
        done = terminated or truncated
    return total_reward, key


if __name__ == "__main__":
    train()
