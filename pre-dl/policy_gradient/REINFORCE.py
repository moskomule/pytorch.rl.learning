from random import choices
import torch
from torch import Tensor
from torch import from_numpy as to_tensor

from policy_gradient.pg_base import PGBase


class REINFORCE(PGBase):
    def __init__(self, env_name, num_episodes=10000, alpha=0.9, gamma=0.9, decay_freq=1000):
        super(REINFORCE, self).__init__(env_name, num_episodes, alpha, gamma, policy="softmax_policy", report_freq=500,
                                        decay_freq=decay_freq)
        self._feature = Tensor(self.action_size, self.obs_size)
        self.actions = range(self.action_size)
        self.min_alpha = 0.1

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
            direction = self.feature(_state, action) - sum(
                [self.softmax @ torch.cat([self.feature(_state, a).unsqueeze(0) for a in self.actions])])
            weight += self.alpha * pow(self.gamma, iter) * reward * direction
            total_reward += reward
            iter += 1
        # update weight
        self.weight = weight
        return total_reward

    def schedule_alpha(self, episode):
        if self.alpha > self.min_alpha and episode % self.decay_freq == 0 and episode != 0:
            self.alpha = self.alpha / (episode / self.decay_freq)

    @property
    def softmax(self):
        numers = torch.exp(Tensor([self.weight @ self.feature(self.state, a) for a in range(self.action_size)]))
        return numers / sum(numers)

    @property
    def softmax_policy(self):
        action = choices(list(range(self.action_size)), weights=self.softmax)[0]
        return action

    def _initialize_weight(self):
        return Tensor(self.obs_size * self.action_size).normal_(0, 1)

    def feature(self, state, action):
        self._feature.zero_()
        self._feature[action] = to_tensor(state).float()
        return self._feature.view(-1)


def main(plot=True, env_name='CartPole-v0'):
    print("start training")
    rf = REINFORCE(env_name)

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
