"""Explore the TF-agents package with the ATARI environment (pag 643)."""

import numpy as np
import time
from tf_agents.environments import suite_gym

n_max_steps = 100

# Load and initialize the environment
env = suite_gym.load(
    environment_name="Breakout-v4",
    max_episode_steps=n_max_steps,
    gym_kwargs={"render_mode":"human"}
    )
print(env.reset())

# Run some random actions
for istep in range(n_max_steps):
    iaction = np.random.randint(0,4) # 0, 1, 2, or 3
    action = env.gym.get_action_meanings()[iaction] # NOOP (do nothing), FIRE (put new coin), RIGHT, LEFT
    print(f"Step {istep:3}: {action:7}")
    # NOTE: game does not starts until the first FIRE
    env.step(iaction)
    # env.render() # NO NEED with gym_kwargs={"render_mode":"human"}
    time.sleep(0.1)

