"""Utilities to explore the Gym package."""

import gym

# Show all environments available
for i, env in enumerate(gym.envs.registry.all()):
    print(i, env.id)
