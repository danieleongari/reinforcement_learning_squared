"""Fixed Q-Value Targets for CartPole (HoML pag 639)."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid message "This TensorFlow binary is optimized with ..."
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt

import gym
import tensorflow as tf 
from tensorflow import keras

def epsilon_greedy_policy(state, epsilon=0): 
    """Given the state, return the action: there is a probability equal to epsilon to return a random action and sample
    the environment, otherwise the action is predicted by the Deep Q-Network model.
    NOTE: if iepisode < episodes_pre_train the model is not trained yet, and the outcome action is anyway random!
    """
    if np.random.rand() < epsilon:
        return np.random.randint(2) # 0 or 1
    else:
        Q_values = train_model.predict(state[np.newaxis]) 
        return np.argmax(Q_values[0])

def play_one_step(env, state, epsilon):
    """Play a single step and append the outcome state to the replay buffer."""
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action) 
    replay_buffer.append((state, action, reward, next_state, done)) 
    return next_state, reward, done, info

def sample_experiences(batch_size):
    """Return a random sample of batch_size experiences, where each element 
    (states, actions, rewards, next_states, dones) is returned as a list of these samples."""
    indices = np.random.randint(len(replay_buffer), size=batch_size) 
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch]) 
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def training_step(batch_size):
    """Use a random sample of experiences to train the model."""
    states, actions, rewards, next_states, dones = sample_experiences(batch_size) 
    next_Q_values = target_model.predict(next_states) # NEW: I'm using a separate target-model for the predictions
    max_next_Q_values = np.max(next_Q_values, axis=1) 
    target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1) # Missing in the book: list to column vector (NOTE: but probably loss_fn can now handle lists too)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = train_model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True) 
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, train_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, train_model.trainable_variables))
    return rewards

# Input environment and settings
env = gym.make("CartPole-v1")

n_episodes = 600            # Episodes to run
n_max_steps = 200           # Max steps in each episode
episodes_pre_train = 50     # Episode when start training
exp_to_remember = 2000      # Experiences stored in the replay_buffer: after that, forget the older ones
episodes_train_target = 50  # NEW: frequency of update for the target model used for predictions
batch_size = 32
discount_factor = 0.95
optimizer = keras.optimizers.Adam(learning_rate=1e-2) # GitHub's ipynb says that 1e-2 is better than 1e-3 
loss_fn = keras.losses.mean_squared_error

# Deep Model
input_shape = [4] # == env.observation_space.shape 
n_outputs = 2 # == env.action_space.n
train_model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])

# NEW: set target model
target_model = keras.models.clone_model(train_model)
target_model.set_weights(train_model.get_weights()) # Initial weights are random, but the same for both "train_model" and "target_model"

# Storage for all agent's experiences (steps), in form of tuples: (obs, action, reward, next_obs, done)
replay_buffer = deque(maxlen=exp_to_remember)

# Set up plot
plt.xlabel('Episode')
plt.ylabel('Return (sum of rewards)')
plt.xlim(0,n_episodes)
plt.ylim(0,n_max_steps)

# Run episodes and training
returns = [] 
for iepisode in range(n_episodes): 
    print(f"Episode {iepisode:3}", end=" ")
    obs = env.reset()
    epsilon = max(1 - iepisode / 500, 0.01) # Probabilty of returning a random action (100% at the beginning)
    for istep in range(200):
        obs, reward, done, info = play_one_step(env, obs, epsilon) 
        if done:
            break 
    print(f"(epsi: {epsilon:.3f}, Return: {istep:3})", end=" - ")
    returns.append(istep) # Remember, in this example rewards are 1.0 for each step

    if iepisode <= episodes_pre_train:
        plot_color = 'blue'
        print("Exploration") # fill up replay buffer, with enough diversity
    else:
        if iepisode % 50 == 0: # NEW: update weights of the target model
            target_model.set_weights(train_model.get_weights())
            plot_color = 'red'
            print("Update weights & Training")
        else:
            plot_color = 'green'
            print("Training")
        
        
        training_step(batch_size)

    # Update plot showing the current return
    if iepisode: # Skip first, which has no previous step
        plt.plot([iepisode, iepisode+1], returns[-2:], color=plot_color)
    plt.pause(0.01)

plt.show()

# Play one final episode with the trained model
# NOTE: the simulation will start when you close the matplotlib window
print("Running episode", end=" ")
obs = env.reset()
valid = True
epsilon = 0 # No need anymore for random exploring
for istep in range(n_max_steps):
    print(".", end="")
    obs, reward, done, info = play_one_step(env, obs, epsilon)
    env.render()
    time.sleep(0.01)
    if valid and done:
        valid=False
        print(f"\nDone after {istep} steps.\n")
        time.sleep(3)
        break 
if valid:
    print(f"\nIt succesfully completed {n_max_steps} steps!\n")

