"""TF-agents package using Deep Q-Network on ATARI game (pag 651)."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid message "This TensorFlow binary is optimized with ..."
import logging
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tf_agents.environments import suite_gym # Don't know why the book uses `suite_atari` instead
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function


class AtariPreprocessingWithAutoFire(AtariPreprocessing):
    """Add to the standard AtariPreprocessing the instruction to play FIRE at the first step."""
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        super().step(1) # FIRE to start
        return obs
    def step(self, action):
        lives_before_action = self.ale.lives()
        obs, rewards, done, info = super().step(action)
        if self.ale.lives() < lives_before_action and not done:
            super().step(1) # FIRE to start after life lost
        return obs, rewards, done, info

class ShowProgress:
    """Show progress each 100 steps as: `{curren_step}/{total_steps}`."""
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

n_max_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames

env = suite_gym.load(
    environment_name="BreakoutNoFrameskip-v4",
    max_episode_steps=n_max_steps,
    gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4],
    #gym_kwargs={"render_mode":"human"} # Delete to exclude original rendering: makes training much slower!
)

tf_env = TFPyEnvironment(env)

preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)

train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε
agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000) # reduce if OOM error

replay_buffer_observer = replay_buffer.add_batch

# Show a log with train metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

# Create the collect driver
collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration

# Collect the initial experiences
initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000) # <=> 80,000 ALE frames # ORIGINAL CODE: 20'000
final_time_step, final_policy_state = init_driver.run()


tf.random.set_seed(9) # chosen to show an example of trajectory at the end of an episode

trajectories, buffer_info = next(iter(replay_buffer.as_dataset(
    sample_batch_size=2,
    num_steps=3,
    single_deterministic_pass=False)))

dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

# Convert the main functions to TF Functions for better performance:
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

# Main loop for training the agent
n_iterations=5000 # ORIGINAL CODE: 50'000, which takes ca. 2h
time_step = None
policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
iterator = iter(dataset)
for iteration in range(n_iterations):
    time_step, policy_state = collect_driver.run(time_step, policy_state)
    trajectories, buffer_info = next(iterator)
    train_loss = agent.train(trajectories)
    print("\r{} loss:{:.5f}".format(
        iteration, train_loss.loss.numpy()), end="")
    if iteration % 1000 == 0:
        log_metrics(train_metrics)

# Show an episode with a trained agent
# TODO: make this program save the trained NN, and another to load and display it.

input("\nModel trained: press a key to make the agent play the game...")

def render_frames(trajectory):
    """Render the current step."""
    tf_env.pyenv.envs[0].render(mode="human")

istep = 0
def print_action(trajectory):
    "Print the action in the current step."
    global istep
    istep+=1
    step = tf_env.pyenv.envs[0].current_time_step()
    iaction = step.step_type
    action = ['NOOP', 'FIRE', 'RIGHT', 'LEFT'][iaction]
    print(f"Step: {istep:5}: {action:7}")

watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[render_frames, print_action],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()