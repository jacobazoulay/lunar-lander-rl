import torch
import gym
import numpy as np
import copy
import time
from matplotlib import pyplot as plt
import os


def save_fig(fig):
    num = 1
    fname = 'figs/plot_'
    while os.path.exists(fname + str(num) + '.png'):
        num += 1
    fig.savefig(fname + str(num) + '.png')


class Lander:
    def __init__(self, lr):
        self.a_space = torch.tensor([0, 1, 2, 3])
        self.a_space_n = 4
        self.obs_space_n = 8

        self.epsilon = 1.0
        self.gamma = 0.9
        self.k = None

        self.Q = QNet(self.obs_space_n, self.a_space_n, lr)
        self.Q_ = QNet(self.obs_space_n, self.a_space_n, lr)

        self.env = gym.make('LunarLander-v2')
        self.observation = torch.as_tensor(self.env.reset()).double()
        self.done = False
        self.reward, self.action, self.action_u = None, None, None

        self.scene_reward = 0.0
        self.scene_rewards = [None]

    def act_optimal(self):
        """
        Given current observation, execute optimal action determined by action-value function Q neural net.
        """
        Q_out = self.Q.forward(self.observation)
        self.action = int(torch.argmax(Q_out))
        self.action_u = Q_out[self.action]

        self.observation, self.reward, self.done, _ = self.env.step(self.action)
        self.observation = torch.as_tensor(self.observation).double()

    def act_random(self):
        """
        Execute random action.
        """
        idx = np.random.randint(0, 4)
        self.action = int(self.a_space[idx])

        Q_out = self.Q.forward(self.observation)
        self.action_u = Q_out[self.action]

        self.observation, self.reward, self.done, _ = self.env.step(self.action)
        self.observation = torch.as_tensor(self.observation).double()

    def e_greedy_step(self, render):
        """
        Given Q neural network, state observations (list of 8 state variables), and epsilon
        execute random policy with epsilon probability. Otherwise act optimally.
        """
        if render == True:
            self.env.render()

        if self.done == True:
            self.observation = torch.as_tensor(self.env.reset()).double()
            self.scene_rewards.append(self.scene_reward)
            self.scene_reward = 0

        if np.random.random_sample() < self.epsilon:
            self.act_random()
        else:
            self.act_optimal()

        self.scene_reward += self.reward
        self.decay_epsilon()

    def train_step(self):
        """
        Perform one step of bellman update using backprop and gradient descent.
        """
        exp_u = torch.max(self.Q_.forward(self.observation))
        Q_target = self.reward + (1 - self.done) * self.gamma * exp_u  # estimated optimal u
        Q_output = self.action_u  # u given by Q(s, a) received when taking action a from s

        criterion = torch.nn.MSELoss()  # squared error loss function
        loss = criterion(Q_output, Q_target)
        self.Q.optimizer.zero_grad()
        loss.backward()
        self.Q.optimizer.step()

    def update_target_net(self):
        """
        Copy main net to target net.
        """
        self.Q_ = copy.deepcopy(self.Q)

    def decay_epsilon(self):
        self.epsilon -= 1 / self.k

    def train(self, k=10000, update_Q_every=100, display_every=1000, render=False):
        self.k = k
        for i in range(self.k):
            if i % update_Q_every == 0:
                self.update_target_net()
            self.e_greedy_step(render)
            self.train_step()
            self.display_info(i, display_every)

    def display_info(self, i, display_every):
        if i % display_every == 0 or i == self.k - 1:
            print(round(100 * i/self.k, 2), '%     Reward: ', self.scene_rewards[-1], '     LR: ', round(self.Q.lr, 6))

    def plot_reward(self):
        self.scene_rewards.pop(0)
        x = range(len(self.scene_rewards))
        y = self.scene_rewards
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_title('Rewards with LR ' + str(round(self.Q.lr, 6)))
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        save_fig(fig)


class QNet:
    def __init__(self, in_n, out_n, lr, l1=64, l2=64):
        # initialize neural network
        self.lr = lr
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


def random_search():
    start = time.time()
    lrs = 10 ** np.linspace(0, -5, 10)
    lrs = [0.0004]
    for lr in lrs:
        lander = Lander(lr)
        lander.train(k=2000000, update_Q_every=100, display_every=50000, render=False)
        lander.plot_reward()
    print('Runtime: ', round(time.time() - start, 2), ' s')


def main():
    # Hypyerparameters include: lr, gamma, net structure, wd
    random_search()


if __name__ == "__main__":
    main()
