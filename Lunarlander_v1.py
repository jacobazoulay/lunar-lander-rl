import torch
import gym
import numpy as np
import random
import copy
import time


class Lander:
    def __init__(self):
        self.a_space = torch.tensor([0, 1, 2, 3])
        self.a_space_n = 4
        self.obs_space_n = 8

        self.Q = QNet(self.obs_space_n, self.a_space_n)
        self.Q_ = QNet(self.obs_space_n, self.a_space_n)

        self.env = gym.make('LunarLander-v2')
        self.observation = torch.as_tensor(self.env.reset()).double()
        self.done = False
        self.reward, self.action, self.action_u = None, None, None

    def act(self):
        self.env.render()
        if self.done == True:
            self.observation = torch.as_tensor(self.env.reset()).double()
        self.action, self.action_u = act_optimal(self.Q.net, self.observation)
        self.observation, self.reward, self.done, _ = self.env.step(self.action)

    def train_step(self):
        # Q_target = reward + (1 - done) * gamma * act_optimal(Q_, observation_next)[1]  # estimated optimal u
        # Q_output = Q(observation)[a_space_disc.index(action)]  # u given by Q(s, a) received when taking action a from s
        # observation = torch.as_tensor(observation_next).double()


class QNet:
    def __init__(self, in_n, out_n, l1=50, l2=50):
        # initialize neural network
        self.lr = 0.9
        self.wd = 0.0
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_n, l1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(l1, l2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(l2, out_n)
                    ).double()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

    def forward(self, obs):
        out = self.net(obs)
        return out


def disc_actions(action_space, bins):
    """
    Given action space and number of bins per action variable,
    return list of discretized action space.
    """
    low_bound = action_space.low[0]  # lower bound of action space is the same for the 4 action variables
    high_bound = action_space.high[0]  # upper bound of action space is the same for the 4 action variables
    a_space_bins = np.round(np.linspace(low_bound, high_bound, num=bins + 2)[1:bins + 1], 2)  # list of evenly spaced actions

    a_space_disc = []
    for i in a_space_bins:
        for j in a_space_bins:
            for k in a_space_bins:
                for m in a_space_bins:
                    a_space_disc.append([i, j, k, m])
    return a_space_disc


def act_optimal(Q, observation):
    """
    Given Q neural network and state observations,
    return best action and action-value determined by Q.
    """
    observation = torch.as_tensor(observation).double()
    Q_out = Q(observation)
    action = int(torch.argmax(Q_out))
    action_u = Q_out[action]
    return action, action_u


def e_greedy_step(Q, observation, epsilon):
    """
    Given Q neural network, state observations (list of 24 state variables), and epsilon
    return random policy with epsilon probability. Otherwise act optimally.
    """
    if np.random.random_sample() < epsilon:
        return random.choice(a_space_disc)
    else:
        return act_optimal(Q, observation)[0]


def prev_test():
    k_1 = 260000    # iterations of Q_ update
    k_2 = 100    # iterations per Q_ update

    reward_sums = []
    reward_sum = 0
    done = False
    for i in range(k_1):
        if i % 100 == 0:
            epsilon = 1 - (i / (k_1 - 1))
            print(epsilon)
        Q_ = copy.deepcopy(Q)
        for j in range(k_2):
            env.render()
            if done:
                reward_sums.append(reward_sum)
                observation = torch.as_tensor(env.reset()).double()
                reward_sum = 0
            action = e_greedy_step(Q_, observation, epsilon)
            observation_next, reward, done, info = env.step(action)
            reward_sum = reward_sum + reward

            Q_target = reward + (1 - done) * gamma * act_optimal(Q_, observation_next)[1]  # estimated optimal u
            Q_output = Q(observation)[a_space_disc.index(action)]  # u given by Q(s, a) received when taking action a from s
            observation = torch.as_tensor(observation_next).double()
            #print(action, ", ", reward)
            #print(Q_output.item(), ", ",  Q_target.item())
            criterion = torch.nn.MSELoss()  # squared error loss function
            loss = criterion(Q_output, Q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    lander = Lander()
    while True:
        lander.act()


if __name__ == "__main__":
    main()