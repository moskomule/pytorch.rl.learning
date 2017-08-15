from random import choices
import torch
from torch import Tensor
from torch import from_numpy as to_tensor
from policy_gradient.pg_base import PGBase


class ActorCritic(PGBase):
    def __init__(self, env_name, num_episodes=10000, alpha=0.9, gamma=0.9, beta=0.1):
        super(ActorCritic, self).__init__(env_name, num_episodes, alpha, gamma, policy="softmax_policy",
                                          report_freq=500, beta=beta)
        self._feature = Tensor(self.action_size, self.obs_size)
        self.actions = range(self.action_size)
        self.min_alpha = 0.1
        self.min_beta = 0.01
        self._state_value_weight = None

    def _loop(self):
        done = False
        total_reward, reward, iter = 0, 0, 0
        self.state = self.env.reset()
        while not done:
            action = self.policy()
            _state, reward, done, _ = self.env.step(action)
            # if _state is terminal, state value is 0
            v = 0 if done else self.state_value(_state)
            delta = reward + self.gamma * v - self.state_value(self.state)
            # \nabla_w v = s, since v = s^{\tim} w
            self.state_value_weight += self.beta * delta * to_tensor(self.state).float()
            # \pi(a) = x^{\top}(a)w, where x is feature and w is weight
            # \nabla\ln\pi(a) = x(a)\sum_b \pi(b)x(b)
            direction = self.feature(_state, action) - sum(
                [self.softmax @ torch.cat([self.feature(_state, a).unsqueeze(0) for a in self.actions])])

            self.weight += self.alpha * pow(self.gamma, iter) * delta * direction
            total_reward += reward
            self.state = _state
            iter += 1
        return total_reward

    def schedule_alpha(self, episode):
        if self.alpha > self.min_alpha and episode % 1000 == 0 and episode != 0:
            self.alpha = self.alpha / (episode / 1000)

    def schedule_beta(self, episode):
        if self.beta > self.min_beta and episode % 1000 == 0 and episode != 0:
            self.beta = self.beta / (episode / 1000)

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

    @property
    def state_value_weight(self):
        if self._state_value_weight is None:
            self._state_value_weight = torch.zeros(self.obs_size)
        return self._state_value_weight

    @state_value_weight.setter
    def state_value_weight(self, x):
        self._state_value_weight = x

    def state_value(self, state):
        state = to_tensor(state).float()
        return state @ self.state_value_weight


def main(plot=True, env_name='CartPole-v0'):
    print("start training")
    ac = ActorCritic(env_name)

    # training
    ac()

    print("testing")
    ac.test(render=False)
    ac.test(render=False)
    ac.test(render=False)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(ac.rewards)
        plt.show()


if __name__ == '__main__':
    main()
