from flax import nnx
import jax.numpy as jnp
import jax


def select_action():
    pass


class DQNAgent:
    def __init__(
        self, network: nnx.Module, action_space_dim: int, gamma: float, epsilon: float
    ):
        self.network = network
        self.epsilon = epsilon
        self.epsilon_decay_reate = 0.0000009
        self.epsilon_mininum = 0.1
        self.epsilon_evaluation = 0.05
        self.action_space_dim = action_space_dim


    def act(self, state: jnp.array, key: jax.random.PRNGKey, training=True) -> tuple[list[int], jax.random.PRNGKey]:
        """
        Selects actions for a batch of states using an epsilon-greedy policy.
        """
        if training:
            actions, key = _act_epsilon_greedy(self.network, state, self.epsilon, key)
            # adjust epsilon
            self.epsilon = max(self.epsilon_mininum, self.epsilon - self.epsilon_decay_reate)
        else:
            actions, key = _act_epsilon_greedy(self.network, state, self.epsilon_evaluation, key)
        return actions.item(), key


class QNetwork(nnx.Module):
    def __init__(self, input_channels, output_dimension, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            input_channels,
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            rngs=rngs,
            kernel_init=nnx.initializers.lecun_uniform(),
        )
        self.conv2 = nnx.Conv(
            32, 64, kernel_size=(4, 4), strides=(2, 2), padding="VALID", rngs=rngs,
            kernel_init=nnx.initializers.lecun_uniform()
        )
        self.conv3 = nnx.Conv(
            64, 64, kernel_size=(3, 3), strides=(1, 1), padding="VALID", rngs=rngs, kernel_init=nnx.initializers.lecun_uniform()
        )
        self.linear1 = nnx.Linear(3136, 512, rngs=rngs, kernel_init=nnx.initializers.lecun_uniform())
        self.linear2 = nnx.Linear(512, output_dimension, rngs=rngs, kernel_init=nnx.initializers.lecun_uniform())

    def __call__(self, x, training: bool = True):
        x = x.astype(jnp.float32) / 255.
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        out = self.linear2(x)
        return out


@nnx.jit
def _act_epsilon_greedy(network: nnx.Module, state: jnp.array, epsilon:jnp.array, key: jax.random.PRNGKey):
    this_state = jnp.expand_dims(state, axis=0)
    key, explore_key, action_key = jax.random.split(key, 3)

    # get q-values from the network for all agents
    q_values = jax.lax.stop_gradient(network(this_state))
    greedy_actions = jnp.argmax(q_values)

    # generate random actions for all agents
    random_actions = jax.random.randint(action_key, shape=(1,), minval=0, maxval=4)

    # decide for each agent whether to explore or exploit
    explore_decisions = jax.random.uniform(explore_key, shape=(1,)) < epsilon

    # select actions based on the exploration decision
    actions = jnp.where(explore_decisions, random_actions, greedy_actions)

    return actions, key