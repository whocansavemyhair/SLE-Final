# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:17:49 2024

@author: jessi
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

# multiple output in notebook without print()
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# Parameters
M = 1.0  # Mass of the cart
m = 0.1  # Mass of the pole
g = -9.8  # Gravity
l = 0.5  # Length of the pole
mu_c = 0.0005  # Friction for the cart
mu_p = 0.000002  # Friction for the pole
delta_t = 0.02  # Time step
actions = [-10, 10]  # Available actions (forces in Newtons)
max_steps_per_episode = 50

# Quantization thresholds
theta_boxes = np.array([-12, -6, -1, 0, 1, 6, 12]) * np.pi / 180  # radians
x_boxes = np.array([-2.4, -0.8, 0.8, 2.4])  # meters
theta_dot_boxes = np.array([-50, 0, 50]) * np.pi / 180  # radians/s
x_dot_boxes = np.array([-0.5, 0, 0.5])  # m/s

# Define state space size
state_space_size = (
    len(theta_boxes) + 1, 
    len(theta_dot_boxes) + 1, 
    len(x_boxes) + 1, 
    len(x_dot_boxes) + 1
)

# Dynamics model
def compute_accelerations(theta, theta_dot, x_dot, F):
    """Compute angular and linear accelerations based on the model."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    theta_ddot = (g * sin_theta + cos_theta * ((-F - m * l * (theta_dot**2) * sin_theta + mu_c * np.sign(x_dot)) / (M + m)) - (mu_p * theta_dot / (m * l))) / (l * (4 / 3 - m * cos_theta**2 / (M + m)))
    x_ddot = (F + m * l * (theta_dot**2 * sin_theta - theta_ddot * cos_theta) - mu_c * np.sign(x_dot)) / (M + m)
    
    return theta_ddot, x_ddot

# Update state using Euler's method
def update_state(state, action):
    """Update the state using Euler integration"""
    theta, theta_dot, x, x_dot = state  # state s = (theta, theta_dot, x, x_dot)
    F = action
    theta_ddot, x_ddot = compute_accelerations(theta, theta_dot, x_dot, F)
    x_dot += delta_t * x_ddot  # update x_dot
    x += delta_t * x_dot  # update x
    theta_dot += delta_t * theta_ddot  # update theta_dot
    theta += delta_t * theta_dot  # update theta
    
    return np.array([theta, theta_dot, x, x_dot])

# Discretize state based on provided thresholds
def discretize_state(state):
    """Discretizes a continuous state into discrete bins"""
    theta, theta_dot, x, x_dot = state
    theta_idx = np.digitize(theta, theta_boxes, right=True) - 1
    theta_dot_idx = np.digitize(theta_dot, theta_dot_boxes, right=True) - 1
    x_idx = np.digitize(x, x_boxes, right=True) - 1
    x_dot_idx = np.digitize(x_dot, x_dot_boxes, right=True) - 1
    
    return (theta_idx, theta_dot_idx, x_idx, x_dot_idx)

# Temporal Difference Learning using TD(0)
def td_learning(alpha=0.7, gamma=0.3, num_episodes=1000, max_steps_per_episode=200):
    """TD(0) Learning Algorithm"""
    value_function = np.zeros(state_space_size)
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = discretize_state(np.random.uniform(low=-np.pi, high=np.pi, size=4))  # Initial state
        total_reward = 0 
        
        for step in range(max_steps_per_episode):
            # Choose action (randomly for now, but could be improved with exploration strategies)
            action = np.random.choice(actions)
            next_state_continuous = update_state([theta_boxes[state[0]], theta_dot_boxes[state[1]], x_boxes[state[2]], x_dot_boxes[state[3]]], action)
            next_state = discretize_state(next_state_continuous)
            reward = 1 if abs(next_state_continuous[0]) <= 12 * np.pi / 180 and abs(next_state_continuous[2]) <= 2.4 else 0
            total_reward += reward

            # TD update
            value_function[state] += alpha * (reward + gamma * value_function[next_state] - value_function[state])
            
            state = next_state  # Move to the next state
        
        # Calculate average reward for this episode
        avg_reward = total_reward / max_steps_per_episode
        episode_rewards.append(avg_reward)
        
        # Print the average reward for this episode
        print(f"Episode {episode + 1}, total Reward: {total_reward}")
            



    return value_function

# Define function for plotting value function w.r.t theta and x for selected theta_dot and x_dot
def plot_value_function(value_function, method_name, fixed_theta_dot_idx, fixed_x_dot_idx):
    """Plots the value function with respect to θ and x."""
    # Slice value function according to fixed indices
    value_function_slice = value_function[:, fixed_theta_dot_idx, :, fixed_x_dot_idx]
    
    # Get the bin edges for plotting
    x_bins = (x_boxes[:-1] + x_boxes[1:]) / 2
    theta_bins = (theta_boxes[:-1] + theta_boxes[1:]) / 2

    plt.figure(figsize=(10, 6))
    plt.imshow(
        value_function_slice,
        extent=[x_bins[0], x_bins[-1], theta_bins[0], theta_bins[-1]],
        aspect='auto',
        origin='lower',
        cmap="coolwarm",
    )
    plt.colorbar(label="Value Function")
    plt.title(f"Value Function with Respect to θ and x ({method_name})")
    plt.xlabel("Cart Position (x) [m]")
    plt.ylabel("Pole Angle (θ) [rad]")
    plt.grid()
    plt.show()

# Implement TD learning
value_function_td = td_learning()

# Plot value function for TD
plot_value_function(value_function_td, "TD Learning", fixed_theta_dot_idx=1, fixed_x_dot_idx=1)

# Parameters for experiments with different discount rates
discount_rates = [0.9, 0.95, 0.99]
num_episodes = 500
trials = 10

# Run simulations for each discount rate
results = []

for gamma in discount_rates:
    avg_rewards = []
    cumulative_rewards_list = []
    success_steps_list = []
    
    for trial in range(trials):
        value_function = np.zeros(state_space_size)  # Reset value function for each trial
        total_rewards = []
        cumulative_rewards = []
        success_steps = []

        state = discretize_state(np.random.uniform(low=-np.pi, high=np.pi, size=4))  # Initial state
        
        for episode in range(num_episodes):
            for _ in range(max_steps_per_episode):
                action = np.random.choice(actions)
                next_state_continuous = update_state([theta_boxes[state[0]], theta_dot_boxes[state[1]], x_boxes[state[2]], x_dot_boxes[state[3]]], action)
                next_state = discretize_state(next_state_continuous)
                reward = 1 if abs(next_state_continuous[0]) <= 12 * np.pi / 180 and abs(next_state_continuous[2]) <= 2.4 else 0

                value_function[state] += 0.1 * (reward + gamma * value_function[next_state] - value_function[state])
                
                total_rewards.append(reward)
                state = next_state  # Move to the next state
            
            cumulative_rewards.append(sum(total_rewards))
            success_steps.append(np.count_nonzero(total_rewards))
    
    results.append((gamma, np.mean(cumulative_rewards), np.mean(success_steps)))

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

#Plot 1: Average total reward per episode vs. number of episodes

array = list(range(1, 1001))  # Create an array from 1 to 1000

for gamma, episode_rewards, _ in results:
    axs[0].plot(array, episode_rewards, label=f'γ={gamma}')  # Use len() to get the size of episode_rewards
axs[0].set_title('Average Total Reward per Episode')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Average Total Reward')
axs[0].legend()
axs[0].grid(True)

# # Plot 2: Average success steps per episode vs. number of episodes
# for gamma, _, success_steps in results:
#     axs[1].plot(np.arange(num_episodes), success_steps, label=f'γ={gamma}')
# axs[1].set_title('Average Success Steps per Episode')
# axs[1].set_xlabel('Episode')
# axs[1].set_ylabel('Average Success Steps')
# axs[1].legend()
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()
