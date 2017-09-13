from math import sqrt
from collections import namedtuple
from random import random, randrange
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

import gym
from tqdm import tqdm
from dl.utils import convert_env, Memory

Transition = namedtuple("Transition", ["state_b", "action", "reward", "state_a", "done"])
logger = getLogger(__name__)
logger.setLevel(INFO)
# stream_handler = StreamHandler()
file_handler = FileHandler("dqn.log")
file_handler.setLevel(0)
# logger.addHandler(stream_handler)
logger.addHandler(file_handler)


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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc(x))
        return F.softmax(self.output(x))

    @staticmethod
    def loss(output, target):
        """
        squared distance
        """
        assert isinstance(output, Variable) and isinstance(target, Variable)
        return torch.sum((output - target) ** 2)


class Agent(object):
    def __init__(self, env: gym.Env, network: nn.Module, gamma, epsilon, final_epsilon, final_exp_step):
        self.env = env
        self.action_size = self.env.action_space.n
        self.net = network(self.action_size)
        self.target_net = network(self.action_size)
        self.gamma = gamma
        self._epsilon = epsilon
        self.epsilon = epsilon
        self._final_epsilon = final_epsilon
        self._final_exp_step = final_exp_step
        self._step_counter = 0
        self.update_target_net()

    def policy(self, state):
        """
        epsilon greedy, state should be np.ndarray
        """
        if random() <= self.epsilon:
            action = randrange(0, self.action_size)
        else:
            state = Variable(torch.from_numpy(np.array(state))).unsqueeze(dim=0)
            action = self.net(state).data.view(-1).max(dim=0)[1].sum()
        logger.debug(f"action: {action}")
        return action

    def parameter_scheduler(self):
        self._step_counter += 1
        if self._step_counter < self._final_exp_step:
            self.epsilon = self._step_counter * (
                self._final_epsilon - self._epsilon) / self._final_exp_step + self._epsilon
        logger.debug(f"ε: {self.epsilon:.2f}")

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
        logger.debug("updated traget network")

    def estimate_value(self, reward, state, done):
        q_hat = self.target_net(state).max(dim=1)[0]
        estim = reward + self.gamma * done * q_hat
        logger.debug(f"reward[0: 3]: {reward[0: 3]}")
        logger.debug(f"done[0: 3]: {done[0: 3]}")
        logger.debug(f"q_hat[0: 3]: {estim[0: 3]}")
        return estim

    def q_value(self, state, action):
        return self.net(state).gather(1, Variable(action.unsqueeze(dim=1)))

    def test(self):
        done = False
        state = self.env.reset()
        while not done:
            self.env.render()
            _state = Variable(torch.from_numpy(np.array(state))).unsqueeze(dim=0)
            action = self.net(_state).data.view(-1).max(dim=0)[1].sum()
            state, reward, done, _ = self.env.step(action)


class Trainer(object):
    def __init__(self, agent: Agent, lr, memory_size, update_freq, batch_size, replay_start):
        self.agent = agent
        self.env = self.agent.env
        self.optimizer = optim.RMSprop(params=self.agent.net.parameters(), lr=lr, momentum=0.95)
        self.memory = Memory(memory_size)
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.replay_start = replay_start
        self._step = 0
        self.warm_up()

    def warm_up(self):
        """
        to populate replay memory
        """
        state_b = self.env.reset()
        for _ in tqdm(range(self.replay_start)):
            action = self.env.action_space.sample()
            state_a, reward, done, _ = self.env.step(action)
            self.memory(Transition(state_b, action, reward, state_a, done))
            state_b = self.env.reset() if done else state_a

    def _train(self):
        """
        neural network part
        """
        self.optimizer.zero_grad()
        batch_state_b, batch_action, batch_reward, batch_state_a, batch_done = self.get_batch()
        target = self.agent.estimate_value(batch_reward, batch_state_a, batch_done)
        loss = self.agent.net.loss(target, self.agent.q_value(batch_state_b, batch_action))
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def train(self, episode):
        logger.info("start training!")
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
                self.agent.parameter_scheduler()

            logger.info(
                    f"ep: {ep}/step: {self._step}/loss: {np.mean(train_loss):.2f}/reward{np.mean(train_reward):.2f}")

    def get_batch(self):
        batch = self.memory.sample(self.batch_size)
        batch_state_b = Variable(
                torch.cat([torch.from_numpy(np.array(m.state_b)).unsqueeze(0) / 255 for m in batch], dim=0))
        batch_action = torch.LongTensor([m.action for m in batch])
        batch_reward = Variable(torch.Tensor([m.reward for m in batch]))
        batch_state_a = Variable(
                torch.cat([torch.from_numpy(np.array(m.state_a)).unsqueeze(0) / 255 for m in batch], dim=0))
        # tensor 0 if done else 1
        batch_done = Variable(1 - torch.Tensor([m.done for m in batch]))
        return batch_state_b, batch_action, batch_reward, batch_state_a, batch_done


def main():
    env = convert_env(gym.make("Pong-v0"))
    agent = Agent(env, DQN, 0.99, 1, 0.1, 100_000)
    trainer = Trainer(agent, 2.5e-4, 100_000, 10_000, 32, 5_000)
    trainer.train(1000)
    torch.save(trainer.agent.net.state_dict(),
               "dqn.wt")


if __name__ == '__main__':
    main()
