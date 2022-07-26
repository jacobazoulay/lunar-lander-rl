# Lunar Lander Reinforcement Learning

This project implements deep Q learning (DQN) to optimize a lunar lander control policy. The LunarLander-v2 environment in OpenAI Gym was used as the testing environment. The main script implements a grid search method in order to determine an efficient learning rate for neural network training. The agent achieved improved results after 1500 episodes (~10 mins).

## Main Script

The Lunarlander.py file contains both the Lander class for the agent and the QNet class for the neural network model.

## Figures
Reward after training for 3000 episodes:
![Reward Over Time](/figs/plot_11.png)

Agent:
![Lander](/figs/lander.png)