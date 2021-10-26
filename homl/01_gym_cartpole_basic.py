"""First example of RL (HoML pag 614)."""
import numpy as np
import gym
import time

def basic_policy(obs):
    """Move the cart left (0) if the pole is pending left, and move right (1) otherwise."""
    angle = obs[2]
    return 0 if angle<0 else 1

# Initialize environment
env = gym.make("CartPole-v1")

nepisode = 10
totals = []
for iepisode in range(nepisode):
    episode_rewards = 0 
    obs = env.reset()
    for istep in range(200):
        action = basic_policy(obs) # env.action_space = Discrete (2), meaning you can only do two moves
        step = env.step(action) # Make the step
        obs = step[0]       # In CartPole: [position, velocity, angle, angular velocity]
        reward = step[1]    # In CartPole always 1 untill Game Over
        done = step[2]      # True if Game Over
        info = step[3]      # Dict, but in CartPole there should not be any
        episode_rewards += reward
        env.render()
        if done:
            print(f"Episode {iepisode:3}: fallen after {episode_rewards:4} steps.")
            time.sleep(1)
            break

    totals.append(episode_rewards)

print("\n--- FINAL STATISTICS ---\n")
print("Number of episodes:", nepisode)
print("Mean/stdev steps:", np.mean(totals), "+/-", np.std(totals))
print("Min/Max steps:", np.min(totals), np.max(totals))