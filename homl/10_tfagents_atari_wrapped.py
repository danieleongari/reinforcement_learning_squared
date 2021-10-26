"""Explore the TF-agents package with the wrapped ATARI environment (pag 643)."""
import numpy as np
import time
import matplotlib.pyplot as plt

from tf_agents.environments import suite_gym # Don't know why the book uses `suite_atari` instead
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

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

n_max_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames

env = suite_gym.load(
    environment_name="BreakoutNoFrameskip-v4",
    max_episode_steps=n_max_steps,
    gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4],
    gym_kwargs={"render_mode":"human"} # Delete to exclude original rendering
)

# Run some random actions
for istep in range(n_max_steps):
    iaction = np.random.randint(2,4) # 2 or 3
    action = env.gym.get_action_meanings()[iaction] # RIGHT, LEFT
    print(f"Step {istep:3}: {action:7}")
    time_step = env.step(iaction)
    obs = time_step.observation
    plt.imshow(obs)
    plt.pause(0.5)
    # >>> Will show both the rendered and wrapped images
    if istep==0:
        time.sleep(2)