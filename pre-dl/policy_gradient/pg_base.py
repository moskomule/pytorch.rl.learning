from base import RLBase
import torch
from functools import reduce
from time import sleep


class PGBase(RLBase):
    def __init__(self, env_name, num_episodes, alpha, gamma, policy, **kwargs):
        super(PGBase, self).__init__(env_name, num_episodes, alpha, gamma, policy, **kwargs)
        self.obs_shape = self.env.observation_space.shape
        self.obs_size = reduce(lambda x, y: x * y, self.obs_shape)
        self.action_size = self.env.action_space.n
        self._feature = torch.Tensor(self.action_size, self.obs_size)
        self._weight = None

    def _loop(self):
        pass

    def policy(self):
        raise NotImplementedError

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
        state = self.env.reset()
        while not done:
            if render:
                self.env.render()
                sleep(interval)
            action = self.policy()
            print(action)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            counter += 1
        print(f"total reward {total_reward} in {counter} steps")
