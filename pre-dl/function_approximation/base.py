import gym
import torch
from torch import Tensor
from torch import from_numpy as to_tensor
import random
from functools import reduce
from time import sleep


class FABase(object):
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
        self.env = gym.make(env_name)

        if not isinstance(self.env.action_space, gym.spaces.Discrete) or \
                not isinstance(self.env.observation_space, gym.spaces.Box):
            raise NotImplementedError("action_space should be discrete and "
                                      "observation_space should be box")

        self.obs_shape = self.env.observation_space.shape
        self.obs_size = reduce(lambda x, y: x * y, self.obs_shape)
        self.action_size = self.env.action_space.n
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state = None
        self._rewards = None
        self._weight = None
        self._policy = policy
        self._feature = torch.Tensor(self.action_size, self.obs_size)
        for k, v in kwargs.items():
            setattr(self, str(k), v)

    def app_q(self, state, action: int):
        """
        approximated q value
        :param state: state in numpy.ndarray
        :param action: action index in int
        :return: approximated q value
        """

        return self.weight @ self.feature(state, action)

    def policy(self) -> int:
        """
        epsilon greedy method
        :return: action (int)
        """
        return getattr(self, self._policy)

    @property
    def epsilon_greedy(self) -> int:
        _epsilon = self.epsilon * (1 - 1 / self.action_size)
        if random.random() > _epsilon:
            action = self.argmax([self.app_q(self.state, a) for a in range(self.action_size)])
        else:
            action = random.randrange(0, self.action_size)
        return action

    def _loop(self):
        """
        Loop in an episode. You need to implement.
        :return: total_reward (list)
        """
        raise NotImplementedError

    def schedule_alpha(self, episode):
        """
        schedule learning rate, this is optional
        :param episode:
        :return:
        """
        pass

    def train(self):
        """
        training the model
        """
        total_reward_list = []
        for episode in range(self.num_episodes):
            self.schedule_alpha(episode)
            total_reward = self._loop()
            total_reward_list.append(total_reward)

            if episode % 100 == 0:
                print(f"episode:{episode} total reward:{total_reward:.2f}")
        self._rewards = total_reward_list

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

    __call__ = train

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
    def rewards(self):
        """
        get reward list
        """
        return self._rewards

    @staticmethod
    def argmax(x):
        if isinstance(x, Tensor):
            return x.max(dim=0)[1][0]
        elif isinstance(x, list):
            return x.index(max(x))
