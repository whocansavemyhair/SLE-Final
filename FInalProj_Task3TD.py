# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:41:22 2024

@author: jessi
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

learning_rate = 0.1
discount_factors = [0.95, 0.99, 0.999]  # Different discount factors to test
EPSILON = 1
epsilon_decay_rate = 0.8
min_epsilon = 0.01
epochs = 1000

theta_bins = [-12 * np.pi / 180, -6 * np.pi / 180, -1 * np.pi / 180, 0, 1 * np.pi / 180, 6 * np.pi / 180, 12 * np.pi / 180]
x_bins = [-2.4, -0.8, 0.8, 2.4]
theta_dot_bins = [-np.inf, -50 * np.pi / 180, 50 * np.pi / 180, np.inf]
x_dot_bins = [-np.inf, -0.5, 0.5, np.inf]

M = 1.0  # Mass of the cart
m = 0.1  # Mass of the pole
g = -9.8  # Gravity
l = 0.5  # Length of the pole
mu_c = 0.0005  # Friction for the cart
mu_p = 0.000002  # Friction for the pole
delta_t = 0.02  # Time step
force = 10  # force

# Q (theta, x, theta_dot, x_dot, action)
Q = {d: np.random.uniform(low=-1, high=1, size=(len(theta_bins)-1, len(x_bins)-1, len(theta_dot_bins)-1, len(x_dot_bins)-1, 2)) for d in discount_factors}

def discretize_state(state):
    theta, x, theta_dot, x_dot = state
    theta_idx = np.digitize(theta, theta_bins) - 1
    x_idx = np.digitize(x, x_bins) - 1
    theta_dot_idx = np.digitize(theta_dot, theta_dot_bins) - 1
    x_dot_idx = np.digitize(x_dot, x_dot_bins) - 1

    theta_idx = np.clip(theta_idx, 0, len(theta_bins) - 2)
    x_idx = np.clip(x_idx, 0, len(x_bins) - 2)
    theta_dot_idx = np.clip(theta_dot_idx, 0, len(theta_dot_bins) - 2)
    x_dot_idx = np.clip(x_dot_idx, 0, len(x_dot_bins) - 2)

    return theta_idx, x_idx, theta_dot_idx, x_dot_idx

def isTerminal(state, time):
    theta, x, theta_dot, x_dot = state
    if abs(theta) > (12*np.pi/180) or abs(x) > 2.4 or time > 200:
        return True
    else:
        return False
    
def extra_reward(state):
    theta, x, theta_dot, x_dot = state
    if abs(theta) < (6*np.pi/180) or abs(x) < 0.8:
        return 10
    else:
        return 0
    
def compute_accelerations(theta, theta_dot, x_dot, F):
    """Compute angular and linear accelerations based on the model."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    theta_ddot = (g * sin_theta + cos_theta * ((-F - m * l * (theta_dot**2) * sin_theta + mu_c * np.sign(x_dot)) / (M + m)) - (mu_p * theta_dot / (m * l))) / (l * (4 / 3 - m * cos_theta**2 / (M + m)))
    x_ddot = (F + m * l * (theta_dot**2 * sin_theta - theta_ddot * cos_theta) - mu_c * np.sign(x_dot)) / (M + m)
    
    return theta_ddot, x_ddot

def update_state(state, force):
    """Update the state using Euler integration"""
    theta, theta_dot, x, x_dot = state  # state s = (theta, theta_dot, x, x_dot)
    F = force
    theta_ddot, x_ddot = compute_accelerations(theta, theta_dot, x_dot, F)
    x_dot += delta_t * x_ddot  # update x_dot
    x += delta_t * x  # update x
    theta_dot += delta_t * theta_ddot  # update theta_dot
    theta += delta_t * theta_dot  # update theta
    
    return np.array([theta, theta_dot, x, x_dot])

def train(discount_factor):
    Q_values = Q[discount_factor]
    rewards = []
    successful_episodes_x = []
    successful_episodes_theta = []
    for episode in range(epochs):
        state_raw = np.array([
            np.random.uniform(-12*np.pi/180, 12*np.pi/180),  # theta
            np.random.uniform(-2.4, 2.4),                   # x
            np.random.uniform(-1*np.pi/180, 1*np.pi/180),                       # theta_dot
            np.random.uniform(-1, 1)                        # x_dot
        ])  # Initial state

        state_raw = np.zeros(4)
        state = discretize_state(state_raw)
        done = False
        total_reward = 0
        time = 0

        while not done:
            # epsilon-greedy
            epsilon = max(epsilon_decay_rate*EPSILON, min_epsilon)
            if random.uniform(0, 1) < epsilon:
                action = np.random.binomial(1, 0.5)
            else:
                action = np.argmax(Q_values[state])
            
            force = action*20 - 10
            next_state_raw = update_state(state_raw,force)
            done = isTerminal(next_state_raw,time)
            reward = extra_reward(next_state_raw) if not done else 0

            # TD(0) update
            next_state = discretize_state(next_state_raw)
            Q_values[state][action] = Q_values[state][action] + learning_rate * (reward + discount_factor * np.max(Q_values[next_state]) - Q_values[state][action])

            state = next_state
            state_raw = next_state_raw
            total_reward += reward
            time += 1

        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Discount factor {discount_factor}: Episode {episode+1}/{epochs}, Average reward: {np.average(rewards[episode-99:episode+1])}")

        # Collect successful episodes state data for plotting
        if total_reward > 0:
            successful_episodes_x.append(state_raw[2])
            successful_episodes_theta.append(state_raw[0])
    
    return rewards, successful_episodes_x, successful_episodes_theta

def plot_results(rewards_by_discount):
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # Plot total reward vs. number of episodes
    for discount_factor, rewards in rewards_by_discount.items():
        axs[0].plot(rewards, label=f'Discount factor: {discount_factor}')
    axs[0].set_title('Total Reward per Episode (Averaged over 10 trials)')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Total Reward')
    axs[0].legend()

    # Plot cumulative reward
    for discount_factor, rewards in rewards_by_discount.items():
        cumulative_rewards = np.cumsum(rewards)
        axs[1].plot(cumulative_rewards, label=f'Discount factor: {discount_factor}')
    axs[1].set_title('Cumulative Reward vs. Episodes')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Cumulative Reward')
    axs[1].legend()

    # Plot x and theta as functions of time for successful episodes
    for discount_factor in rewards_by_discount.keys():
        successful_episodes_x, successful_episodes_theta = zip(*[train(discount_factor)[1:3]])
        for x, theta in zip(successful_episodes_x, successful_episodes_theta):
            axs[2].plot(x, label=f'Successful x (discount factor: {discount_factor})')
            axs[2].plot(theta, label=f'Successful theta (discount factor: {discount_factor})')
    
    axs[2].set_title('x and θ as Functions of Time for Successful Episodes')
    axs[2].set_xlabel('Time Steps')
    axs[2].set_ylabel('State Variables')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

# Train and collect results
rewards_by_discount = {d: train(d)[0] for d in discount_factors}

# Plot the results
plot_results(rewards_by_discount)
