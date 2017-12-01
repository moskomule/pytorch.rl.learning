import gym
import torch
from torch import Tensor
from torch import from_numpy as to_tensor
import random
from functools import reduce
from time import sleep

from predl import RLBase


class FABase(RLBase):
    def __init__(self, env_name, num_episodes, alpha, gamma, epsilon, policy, **kwargs):
        """
        base class for RL using lookup table
        :param env_name: name of environment, currently environments whose observation space is Box and action space is
         Discrete are supported. see https://github.com/openai/gym/wiki/Table-of-environments
        :param num_episodes: number of episode for training
        :param alpha:
        :param gamma:
        :param epsilon:
        :param kwargs: other arguments.
        """
        super(FABase, self).__init__(env_name, num_episodes, alpha, gamma, policy, epsilon=epsilon, **kwargs)

        if not isinstance(self.env.action_space, gym.spaces.Discrete) or \
                not isinstance(self.env.observation_space, gym.spaces.Box):
            raise NotImplementedError("action_space should be discrete and "
                                      "observation_space should be box")

        self.obs_shape = self.env.observation_space.shape
        self.obs_size = reduce(lambda x, y: x * y, self.obs_shape)
        self.action_size = self.env.action_space.n
        self._feature = torch.Tensor(self.action_size, self.obs_size)
        self._weight = None

    def app_q(self, state, action: int):
        """
        approximated q value
        :param state: state in numpy.ndarray
        :param action: action index in int
        :return: approximated q value
        """

        return self.weight @ self.feature(state, action)

    def test(self, render=False, interval=0.1):
        """
        testing the trained model
        :param init_state: the initial state
        """
        done = False
        total_reward, reward, counter = 0, 0, 0
        state = self.env.reset()
        while not done:
            if render:
                self.env.render()
                sleep(interval)
            action = self.argmax([self.app_q(state, a) for a in range(self.action_size)])
            print(action)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            counter += 1
        print(f"total reward {total_reward} in {counter} steps")

    @property
    def weight(self):
        if self._weight is None:
            self._weight = Tensor(self.obs_size * self.action_size).normal_(0, 1)
        return self._weight

    @weight.setter
    def weight(self, x):
        self._weight = x

    def feature(self, state, action):
        """
        create feature from (state, action)
        """
        self._feature.zero_()
        self._feature[action] = to_tensor(state).float()
        return self._feature.view(-1)

    @property
    def epsilon_greedy(self) -> int:
        _epsilon = self.epsilon * (1 - 1 / self.action_size)
        if random.random() > _epsilon:
            action = self.argmax([self.app_q(self.state, a) for a in range(self.action_size)])
        else:
            action = random.randrange(0, self.action_size)
        return action
