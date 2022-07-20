import torch
import gym
import numpy as np
import copy


class Lander:
    def __init__(self):
        self.a_space = torch.tensor([0, 1, 2, 3])
        self.a_space_n = 4
        self.obs_space_n = 8

        self.epsilon = 0.5
        self.gamma = 0.9

        self.Q = QNet(self.obs_space_n, self.a_space_n)
        self.Q_ = QNet(self.obs_space_n, self.a_space_n)

        self.env = gym.make('LunarLander-v2')
        self.observation = torch.as_tensor(self.env.reset()).double()
        self.done = False
        self.reward, self.action, self.action_u = None, None, None

    def act_optimal(self):
        """
        Given current observation, execute optimal action determined by action-value function Q neural net.
        """
        self.env.render()
        if self.done == True:
            self.observation = torch.as_tensor(self.env.reset()).double()

        Q_out = self.Q.forward(self.observation)
        self.action = int(torch.argmax(Q_out))
        self.action_u = Q_out[self.action]

        self.observation, self.reward, self.done, _ = self.env.step(self.action)
        self.observation = torch.as_tensor(self.observation).double()

    def act_random(self):
        """
        Execute random action.
        """
        self.env.render()
        if self.done == True:
            self.observation = torch.as_tensor(self.env.reset()).double()

        idx = np.random.randint(0, 4)
        self.action = int(self.a_space[idx])

        Q_out = self.Q.forward(self.observation)
        self.action_u = Q_out[self.action]

        self.observation, self.reward, self.done, _ = self.env.step(self.action)
        self.observation = torch.as_tensor(self.observation).double()

    def e_greedy_step(self):
        """
        Given Q neural network, state observations (list of 8 state variables), and epsilon
        execute random policy with epsilon probability. Otherwise act optimally.
        """
        if np.random.random_sample() < self.epsilon:
            self.act_random()
        else:
            self.act_optimal()

    def train_step(self):
        exp_u = torch.max(self.Q_.forward(self.observation))
        Q_target = self.reward + (1 - self.done) * self.gamma * exp_u  # estimated optimal u
        Q_output = self.action_u  # u given by Q(s, a) received when taking action a from s

        criterion = torch.nn.MSELoss()  # squared error loss function
        loss = criterion(Q_output, Q_target)
        self.Q.optimizer.zero_grad()
        loss.backward()
        self.Q.optimizer.step()

    def update_target_net(self):
        self.Q_ = copy.deepcopy(self.Q)


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


def main():
    lander = Lander()
    k = 1000
    for i in range(k):
        if k % 100 == 0:
            lander.update_target_net()
        lander.e_greedy_step()
        lander.train_step()


if __name__ == "__main__":
    main()
