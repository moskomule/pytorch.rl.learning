from functools import reduce
from time import sleep
import gym

from predl import RLBase


class PGBase(RLBase):
    def __init__(self, env_name, num_episodes, alpha, gamma, policy, **kwargs):
        """
        base class for RL using policy gradient
        :param env_name: name of environment, currently environments whose observation space is Box and action space is
         Discrete are supported. see https://github.com/openai/gym/wiki/Table-of-environments
        :param num_episodes:
        :param alpha:
        :param gamma:
        :param policy:
        :param kwargs:
        """
        super(PGBase, self).__init__(env_name, num_episodes, alpha, gamma, policy, **kwargs)
        if not isinstance(self.env.action_space, gym.spaces.Discrete) or \
                not isinstance(self.env.observation_space, gym.spaces.Box):
            raise NotImplementedError("action_space should be discrete and "
                                      "observation_space should be box")
        self.obs_shape = self.env.observation_space.shape
        self.obs_size = reduce(lambda x, y: x * y, self.obs_shape)
        self.action_size = self.env.action_space.n
        self._feature = None
        self._weight = None

    def _loop(self):
        done = False
        total_reward, reward = 0, 0
        self.state = self.env.reset()
        action = self.policy()
        while not done:
            _state, reward, done, _ = self.env.step(action)
            _action = self.argmax([self.app_q(_state, a) for a in range(self.action_size)])
            q = self.app_q(self.state, action)
            target = reward + self.gamma * self.app_q(_state, _action)
            # todo use autograd instead
            self.weight -= self.alpha * (target - q) * self.feature(self.state, action)
            total_reward += reward
            self.state = _state
            action = _action
        return total_reward

    @property
    def weight(self):
        if self._weight is None:
            self._weight = self._initialize_weight()
        return self._weight

    def _initialize_weight(self):
        raise NotImplementedError

    @weight.setter
    def weight(self, x):
        self._weight = x

    def feature(self, state, action):
        """
        create feature from (state, action)
        """
        raise NotImplementedError

    def test(self, render=False, interval=0.1):
        done = False
        total_reward, reward, counter = 0, 0, 0
        self.state = self.env.reset()
        while not done:
            if render:
                self.env.render()
                sleep(interval)
            action = self.policy()
            print(action)
            self.state, reward, done, _ = self.env.step(action)
            total_reward += reward
            counter += 1
        print(f"total reward {total_reward} in {counter} steps")
