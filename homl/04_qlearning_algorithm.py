"""Q-Learning implemented (HoML pag 631)."""

import numpy as np

transition_probabilities = [ # shape=[s, a, s']
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]], 
    [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
    [None, [0.8, 0.1, 0.1], None]
]

rewards = [ # shape=[s, a, s']
    [[+10, 0, 0], [0, 0, 0], [0, 0, 0]], 
    [[0, 0, 0], [0, 0, 0], [0, 0, -50]], 
    [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]
]

possible_actions = [
    [0, 1, 2], 
    [0, 2], 
    [1]
]

# Initialize Q-Values to 0 for possible actions
Q_values = np.full((3, 3), -np.inf) # -np.inf for impossible actions 
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0 # for all possible actions

alpha0 = 0.05 # initial learning rate 
decay = 0.005 # learning rate decay 
gamma = 0.90 # discount factor
state = 0 # initial state

def step(state, action):
    probas = transition_probabilities[state][action] 
    next_state = np.random.choice([0, 1, 2], p=probas) 
    reward = rewards[state][action][next_state] 
    return next_state, reward

def exploration_policy(state):
    return np.random.choice(possible_actions[state])

for iteration in range(10000):
    action = exploration_policy(state)
    next_state, reward = step(state, action) # Note: the step is made without the explicit knowledge of the tranisiton_probbilities
    next_value = np.max(Q_values[next_state])
    alpha = alpha0 / (1 + iteration * decay)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + gamma * next_value) 
    state = next_state
    if iteration%200==0:
        print(f"Iteration {iteration:3} - Q-value[0,0] = {Q_values[0, 0]:5}")

# Print highest Q-value at each step
print()
for istep, step in enumerate(Q_values):
    print(f"Step {istep}, action with highest Q-Value: {np.argmax(step)}")