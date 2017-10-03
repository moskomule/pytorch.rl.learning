from collections import namedtuple
from random import random, randrange
from logging import getLogger, FileHandler, INFO, DEBUG

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

import gym
from tqdm import tqdm
from utils import convert_env, Memory

cuda = torch.cuda.is_available()


def variable(t: torch.Tensor, **kwargs):
    if cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


Transition = namedtuple("Transition", ["state_b", "action", "reward", "state_a", "done"])
logger = getLogger(__name__)
file_handler = FileHandler("dqn.log")
file_handler.setLevel(DEBUG)
logger.setLevel(INFO)
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

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc(x))
        x = self.output(x)
        return x

    @staticmethod
    def loss(output, target, *args):
        """
        squared distance
        """
        assert isinstance(output, Variable) and isinstance(target, Variable)
        # return torch.mean(torch.sum((output - target).clamp(-1, 1) ** 2, dim=1))
        return F.smooth_l1_loss(output, target, size_average=False)


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
        if cuda:
            self.net.cuda()
            self.target_net.cuda()
        self.update_target_net()

    def policy(self, state):
        """
        epsilon greedy, state should be np.ndarray
        """
        if random() <= self.epsilon:
            action = randrange(0, self.action_size)
        else:
            state = variable(torch.from_numpy(np.array(state))).unsqueeze(0)
            action = self.net(state).data.view(-1).max(dim=0)[1].sum()
        return action

    def parameter_scheduler(self):
        self._step_counter += 1
        if self._step_counter < self._final_exp_step:
            self.epsilon = self._step_counter * (
                self._final_epsilon - self._epsilon) / self._final_exp_step + self._epsilon
        logger.debug(f"ε: {self.epsilon:.2f}")

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False
        logger.info("updated traget network")

    def estimate_value(self, reward, state, done):
        q_hat = self.target_net(variable(state)).max(dim=1)[0]
        estim = variable(reward) + self.gamma * variable(done) * q_hat
        return estim

    def q_value(self, state, action):
        return self.net(variable(state)).gather(1, variable(action.unsqueeze(dim=1)))

    def render(self):
        done = False
        state = self.env.reset()
        while not done:
            self.env.render()
            _state = variable(torch.from_numpy(np.array(state))).unsqueeze(dim=0)
            action = self.net(_state).data.view(-1).max(dim=0)[1].sum()
            state, reward, done, _ = self.env.step(action)


class Trainer(object):
    def __init__(self, agent: Agent, lr, memory_size, update_freq, batch_size, replay_start):
        self.agent = agent
        self.env = self.agent.env
        self.optimizer = optim.Adam(params=self.agent.net.parameters(), lr=lr)
        self.memory = Memory(memory_size)
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.replay_start = replay_start
        self._step = 0
        self._warmed = False

    def warm_up(self):
        """
        to populate replay memory
        """
        logger.info("warming up")
        state_b = self.env.reset()
        for _ in tqdm(range(self.replay_start)):
            action = self.env.action_space.sample()
            state_a, reward, done, _ = self.env.step(action)
            self.memory(Transition(state_b, action, reward, state_a, done))
            state_b = self.env.reset() if done else state_a

    def _nn_part(self):
        """
        neural network part
        """
        self.optimizer.zero_grad()
        batch_state_b, batch_action, batch_reward, batch_state_a, batch_done = self.get_batch()
        target = self.agent.estimate_value(batch_reward, batch_state_a, batch_done)
        q_value = self.agent.q_value(batch_state_b, batch_action)
        if self._step % 1000 == 0:
            logger.info(f"target: {target.data.view(-1).tolist()[:5]}")
            logger.info(f"q_value: {q_value.data.view(-1).tolist()[:5]}")
            logger.info(f"epsilon: {self.agent.epsilon}")
        loss = self.agent.net.loss(q_value, target)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def train(self, episode):
        if not self._warmed:
            self.warm_up()
        logger.info("start training!")
        for ep in range(episode):
            done = False
            state_b = self.env.reset()
            train_loss = []
            train_reward = []
            actions = []
            while not done:
                self._step += 1
                action = self.agent.policy(state_b)
                state_a, reward, done, _ = self.env.step(action)
                self.memory(Transition(state_b, action, reward, state_a, done))
                state_b = state_a
                train_loss.append(self._nn_part())
                if self._step % self.update_freq == 0:
                    self.agent.update_target_net()
                train_reward.append(reward)
                actions.append(action)
                self.agent.parameter_scheduler()

            logger.info(f"ep: {ep+1:>5}/step: {self._step:>6}/"
                        f"loss: {np.mean(train_loss):>7.2f}/reward: {sum(train_reward):.2f}/size: {len(train_loss)}")
            if ep % 50 == 0:
                logger.info(f"actions: {actions}")

    def get_batch(self):
        batch = self.memory.sample(self.batch_size)
        batch_state_b = torch.cat([torch.from_numpy(np.array(m.state_b)).unsqueeze(0) / 255 for m in batch], dim=0)
        batch_action = torch.LongTensor([m.action for m in batch])
        batch_reward = torch.Tensor([m.reward for m in batch])
        batch_state_a = torch.cat([torch.from_numpy(np.array(m.state_a)).unsqueeze(0) / 255 for m in batch], dim=0)
        # tensor 0 if done else 1
        batch_done = 1 - torch.Tensor([m.done for m in batch])
        return batch_state_b, batch_action, batch_reward, batch_state_a, batch_done


def main(is_debug=False):
    env = convert_env(gym.make("Pong-v4"))
    if is_debug:
        agent = Agent(env, DQN, 0.99, 1, 0.1, 10_000)
        trainer = Trainer(agent, 2.5e-4, 10_000, 10_000, 16, 5_000)
    else:
        agent = Agent(env, DQN, 0.99, 1, 0.1, 1_000_000)
        trainer = Trainer(agent, 2.5e-4, 1_000_000, 10_000, 32, 50_000)
    trainer.train(50000)
    torch.save(trainer.agent.net.state_dict(),
               "dqn.wt")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    main(args.debug)
