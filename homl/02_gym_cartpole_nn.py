"""Neural Network policy for CartPole (HoML pag 617)."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid message "This TensorFlow binary is optimized with ..."
import numpy as np
import time

import gym
import tensorflow as tf 
from tensorflow import keras

def play_one_step(env, obs, model, loss_fn): 
    """Play a single step of the CartPole environment: 
    giving the previou step's obs, return the next step's obs.
    In the meanwhile it computes also the gradient of the loss with respect to the model's parameters.
    """
    with tf.GradientTape() as tape:
            left_proba = model(obs[np.newaxis])                             # np.newaxis turns a n-vector into a nx1 matrix
            action = (tf.random.uniform([1, 1]) > left_proba)
            y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
            loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, _ = env.step(int(action[0, 0].numpy())) 
    return obs, reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn): 
    """Given a number of episodes to play, return for each a list of rewards and gradients (one for each step).
    """

    # Pre-allocate list of lists
    all_rewards = [ [None for _ in range(n_max_steps)] for _ in range(n_episodes) ]
    all_grads = [ [None for _ in range(n_max_steps)] for _ in range(n_episodes) ]

    for iepisode in range(n_episodes):
        obs = env.reset()
        for istep in range(n_max_steps):
            obs, all_rewards[iepisode][istep], done, all_grads[iepisode][istep] = play_one_step(env, obs, model, loss_fn) 
            if done:
                # Remove unused slots
                all_rewards[iepisode] = all_rewards[iepisode][:istep]
                all_grads[iepisode] = all_grads[iepisode][:istep]
                break
    return all_rewards, all_grads

def discount_rewards(rewards, discount_factor): 
    """Given a list of rewards, start from the penultimate and go back discounting them all."""
    discounted = np.array(rewards)
    for istep in range(len(rewards) - 2, -1, -1):
        discounted[istep] += discounted[istep + 1] * discount_factor 
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor): 
    """Given a list of rewards for a list of episodes, discount and normalize (by the overall mean and std) them."""
    all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards] 
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


# Input environment and settings
env = gym.make("CartPole-v1")

n_iterations = 100          # Training iterations
n_episodes_per_update = 10  # Episodes for each training iteration
n_max_steps = 200           # Max steps in each episode
discount_factor = 0.95      # Typically 0.90-0.95
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.binary_crossentropy

# Model: Given four inputs from CartPole, return a single value: the probability of going left
n_inputs = 4 # == env.observation_space.shape[0]
model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])

# Training iterations
for iteration in range(n_iterations):
    print(f'Training iteration {iteration:3}', end=' - ')
    all_rewards, all_grads = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)
    all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)
    all_returns =  [ sum(l) for l in all_rewards ]

    # Scale each gradient by the discounted reward, and apply mean gradients for each model's parameter (variable)
    all_mean_grads = []
    for ivariable in range(len(model.trainable_variables)):
        grads = []
        for iepisode, final_rewards in enumerate(all_final_rewards): 
            for istep, final_reward in enumerate(final_rewards):
                grads.append(final_reward * all_grads[iepisode][istep][ivariable])
        mean_grads = tf.reduce_mean(grads, axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
    print(f"Mean return: {round(np.mean(all_returns), 2):5} Max/Min return: {np.max(all_returns):5} {np.min(all_returns):5}")

# Play one final episode with the trained model
print("Running episode", end=" ")
obs = env.reset()
valid = True
for istep in range(n_max_steps):
    print(".", end="")
    obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
    env.render()
    time.sleep(0.01)
    if valid and done:
        valid=False
        print(f"\nDone after {istep} steps.\n")
        # break 
if valid:
    print(f"\nIt succesfully completed {n_max_steps} steps!\n")

