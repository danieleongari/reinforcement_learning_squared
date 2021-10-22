"""Q-Value Iteration Algorithm implemented (HoML pag 628)."""

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

gamma = 0.90 # the discount factor

# Itearate
for iteration in range(50): 
    Q_prev = Q_values.copy()
    for s in range(3):
        for a in possible_actions[s]: 
            Q_values[s, a] = np.sum([ transition_probabilities[s][a][sp] * ( # Note: it does requires to know the trainsition probabilities explicitly
                rewards[s][a][sp] + gamma * np.max(Q_prev[sp])) for sp in range(3)])
    
    print(f"Iteration {iteration:3} - Q-value[0,0] = {Q_values[0, 0]:5}")

# Print highest Q-value at each step
print()
for istep, step in enumerate(Q_values):
    print(f"Step {istep}, action with highest Q-Value: {np.argmax(step)}")



