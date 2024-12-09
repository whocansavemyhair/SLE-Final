# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:14:23 2024

@author: jessica ahner
"""

#SLE Final Project Task 2
# Jessica Ahner, Yeonsoo Lim, & Yichen Lin
import numpy as np

# Gridworld parameters
states = list(range(1, 17))  # States 1 to 16
actions = ['up', 'down', 'right', 'left']
gamma = 1.0  # Undiscounted task
theta = 1e-5  # Threshold for policy evaluation convergence

# Define the transition dynamics
def get_next_state_reward(state, action):
    """Get the next state and reward for a given state and action."""
    if state in [1, 16]:  # Terminal states
        return state, 0

    row, col = divmod(state - 1, 4)  # Convert state to grid position
    if action == 'up':
        next_row, next_col = max(row - 1, 0), col
    elif action == 'down':
        next_row, next_col = min(row + 1, 3), col
    elif action == 'left':
        next_row, next_col = row, max(col - 1, 0)
    elif action == 'right':
        next_row, next_col = row, min(col + 1, 3)
    
    next_state = next_row * 4 + next_col + 1  # Convert back to state
    reward = -1  # Reward for all transitions
    return next_state, reward

# Initialize value function and policy
V = np.zeros(len(states))  # Value function
policy = {s: np.random.choice(actions) for s in states if s not in [1, 16]}  # Random policy

# Policy Evaluation
def policy_evaluation(policy, V):
    """Evaluate a policy to find the value function."""
    while True:
        delta = 0
        for state in states:
            if state in [1, 16]:  # Skip terminal states
                continue
            v = V[state - 1]
            new_v = 0
            for action in actions:
                prob = 1 / len(actions)  # Equiprobable random policy
                next_state, reward = get_next_state_reward(state, action)
                new_v += prob * (reward + gamma * V[next_state - 1])
            V[state - 1] = new_v
            delta = max(delta, abs(v - new_v))
        if delta < theta:
            break
    return V

# Policy Improvement
def policy_improvement(V):
    """Improve the policy based on the value function."""
    policy_stable = True
    new_policy = {}
    for state in states:
        if state in [1, 16]:  # Skip terminal states
            continue
        action_values = []
        for action in actions:
            next_state, reward = get_next_state_reward(state, action)
            action_values.append(reward + gamma * V[next_state - 1])
        best_action = actions[np.argmax(action_values)]
        if state in policy and best_action != policy[state]:
            policy_stable = False
        new_policy[state] = best_action
    return new_policy, policy_stable

# Policy Iteration
def policy_iteration():
    """Perform Policy Iteration to find the optimal policy and value function."""
    global policy, V
    while True:
        V = policy_evaluation(policy, V)
        policy, policy_stable = policy_improvement(V)
        if policy_stable:
            break
    return policy, V

# Run the algorithm
optimal_policy, optimal_value_function = policy_iteration()

# Reshape results for display
optimal_value_function = np.round(optimal_value_function, 2).reshape((4, 4))
optimal_policy_grid = np.full((4, 4), '', dtype=object)
for state, action in optimal_policy.items():
    row, col = divmod(state - 1, 4)
    optimal_policy_grid[row, col] = action

# Display results
print("Optimal Value Function:\n", optimal_value_function)
print("\nOptimal Policy:\n", optimal_policy_grid)

