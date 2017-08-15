from policy_gradient.REINFORCE import REINFORCE

import torch
from torch import Tensor
from torch import from_numpy as to_tensor


class RFBaseline(REINFORCE):
    def __init__(self, env_name, num_episodes=10000, alpha=0.9, gamma=0.9, beta=0.1):
        super(RFBaseline, self).__init__(env_name, num_episodes, alpha, gamma)
        self.beta = beta
        self._feature = Tensor(self.action_size, self.obs_size)
        self._baseline_weight = None
        self.actions = range(self.action_size)

    def _loop(self):
        done = False
        total_reward, reward, iter = 0, 0, 0
        self.state = self.env.reset()
        weight = self.weight
        while not done:
            action = self.policy()
            _state, reward, done, _ = self.env.step(action)
            # use current weight to generate an episode
            # \pi(a) = x^{\top}(a)w, where x is feature and w is weight
            # \nabla\ln\pi(a) = x(a)\sum_b \pi(b)x(b)
            delta = reward - self.state_value(_state)
            self.baseline_weight += self.beta * delta * to_tensor(_state).float()
            direction = self.feature(_state, action) - sum(
                [self.softmax @ torch.cat([self.feature(_state, a).unsqueeze(0) for a in self.actions])])
            weight += self.alpha * pow(self.gamma, iter) * delta * direction
            total_reward += reward
            iter += 1
        # update weight
        self.weight = weight
        return total_reward

    def _initialize_weight(self):
        return Tensor(self.obs_size * self.action_size).normal_(0, 1)

    def feature(self, state, action):
        self._feature.zero_()
        self._feature[action] = to_tensor(state).float()
        return self._feature.view(-1)

    @property
    def baseline_weight(self):
        if self._baseline_weight is None:
            self._baseline_weight = torch.zeros(self.obs_size)
        return self._baseline_weight

    @baseline_weight.setter
    def baseline_weight(self, x):
        self._baseline_weight = x

    def state_value(self, state):
        state = to_tensor(state).float()
        return state @ self.baseline_weight


def main(plot=True, env_name='CartPole-v0'):
    print("start training")
    rf = RFBaseline(env_name, num_episodes=int(1e5))

    # training
    rf()

    print("testing")
    rf.test(render=False)
    rf.test(render=False)
    rf.test(render=False)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(rf.rewards)
        plt.show()


if __name__ == '__main__':
    main()
