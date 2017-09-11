from collections import namedtuple
from random import random, randrange
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

import gym
from tqdm import tqdm
from dl.utils import convert_env, Memory

Transition = namedtuple("Transition", ["state_b", "action", "reward", "state_a", "done"])


class DQN(nn.Module):
    def __init__(self, output_size: int):
        """
        naïve implementation of Deep Q Network proposed in Nature 2015
        naïve: showing a lack of experience, wisdom, or judgement
        :param output_size: actions
        """
        super(DQN, self).__init__()
        self.feature = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(inplace=True))
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.output = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc(x))
        return self.output(x)

    @staticmethod
    def loss(output, target):
        """
        squared distance
        """
        assert isinstance(output, Variable) and isinstance(target, Variable)
        return torch.sum((output - target) ** 2)


class Agent(object):
    def __init__(self, env: gym.Env, network: nn.Module, gamma, epsilon, final_epsilon):
        self.env = env
        self.action_size = self.env.action_space.n
        self.net = network(self.action_size)
        self.target_net = network(self.action_size)
        self.update_target_net()
        self.gamma = gamma
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon

    def policy(self, state):
        """
        epsilon greedy, state should be np.ndarray
        """
        if random() < self.epsilon:
            action = randrange(0, self.action_size)
        else:
            state = Variable(torch.from_numpy(np.array(state))).unsqueeze(dim=0)
            action = self.net(state).data.view(-1).max(dim=0)[1].sum()
        return action

    def parameter_scheduler(self):
        self.epsilon /= 2
        if self.epsilon < self.final_epsilon:
            self.epsilon = self.final_epsilon

    def update_target_net(self):
        logging.info("updated traget network")
        self.target_net.load_state_dict(self.net.state_dict())

    def estimate_value(self, reward, state, done):
        q_hat = self.target_net(state).max(dim=1)[0]
        print(q_hat)
        return reward + self.gamma * done * q_hat

    def q_value(self, state, action):
        return self.net(state).gather(1, Variable(action.view(-1, 1)))

    def test(self):
        done = False
        while not done:
            pass


class Trainer(object):
    def __init__(self, agent: Agent, lr, memory_size, update_freq, batch_size, replay_start):
        self.agent = agent
        self.env = self.agent.env
        self.optimizer = optim.RMSprop(params=agent.net.parameters(), lr=lr)
        self.memory = Memory(memory_size)
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.replay_start = replay_start
        self._step = 0
        self.warm_up()

    def warm_up(self):
        state_b = self.env.reset()
        for _ in tqdm(range(self.replay_start)):
            action = self.env.action_space.sample()
            state_a, reward, done, _ = self.env.step(action)
            self.memory(Transition(state_b, action, reward, state_a, done))
            state_b = self.env.reset() if done else state_a

    def _train(self):
        """
        neural network part
        :return:
        """
        self.optimizer.zero_grad()
        batch_state_b, batch_action, batch_reward, batch_state_a, batch_done = self.get_batch()
        target = self.agent.estimate_value(batch_reward, batch_state_a, batch_done)
        loss = self.agent.net.loss(target, self.agent.q_value(batch_state_b, batch_action))
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def train(self, episode):
        for ep in range(episode):
            done = False
            state_b = self.env.reset()
            train_loss = []
            train_reward = []
            while not done:
                self._step += 1
                action = self.agent.policy(state_b)
                state_a, reward, done, _ = self.env.step(action)
                self.memory(Transition(state_b, action, reward, state_a, done))
                state_b = state_a
                train_loss.append(self._train())
                if self._step % self.update_freq == 0:
                    self.agent.update_target_net()
                train_reward.append(reward)

                if self._step % 100 == 0:
                    logging.info(f"step: {self._step}/{np.mean(train_loss):.2f}/{np.mean(train_reward):.2f}")
                    logging.debug(f">>ε:{self.agent.epsilon}")
            self.agent.parameter_scheduler()
        return self.agent

    def get_batch(self):
        batch = self.memory.sample(self.batch_size)
        batch_state_b = Variable(
                torch.cat([torch.from_numpy(np.array(m.state_b)).unsqueeze(0) / 255 for m in batch], dim=0))
        batch_action = torch.LongTensor([m.action for m in batch])
        batch_reward = Variable(torch.Tensor([m.reward for m in batch]))
        batch_state_a = Variable(
                torch.cat([torch.from_numpy(np.array(m.state_a)).unsqueeze(0) / 255 for m in batch], dim=0))
        batch_done = Variable(torch.Tensor([m.done for m in batch]))
        return batch_state_b, batch_action, batch_reward, batch_state_a, batch_done


def main():
    env = convert_env(gym.make("Pong-v0"))
    agent = Agent(env, DQN, 0.99, 1, 0.1)
    trainer = Trainer(agent, 2.5e-4, 1_000_000, 5_000, 32, 50_000)
    trainer.train(100)


if __name__ == '__main__':
    main()
