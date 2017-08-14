from function_approximation.base import FABase
import matplotlib.pyplot as plt


class QLearning(FABase):
    def __init__(self, env_name, num_episodes=50000, alpha=0.9, gamma=0.9, epsilon=1e-1, min_alpha=1e-3):
        super(QLearning, self).__init__(env_name, num_episodes, alpha, gamma, epsilon, min_alpha=min_alpha)

    def _loop(self):
        done = False
        total_reward, reward = 0, 0
        self.state = self.env.reset()
        action = self.policy()
        while not done:
            _state, reward, done, _ = self.env.step(action)
            q = self.app_q(self.state, action)
            target = reward + self.gamma * max([self.app_q(_state, a) for a in range(self.action_size)])
            # todo use autograd instead
            self.weight -= self.alpha * (target - q) * self.feature(self.state, action)
            total_reward += reward
            self.state = _state
        return total_reward

    def schedule_alpha(self, episode):
        if self.alpha > self.min_alpha and episode % 1000 == 0 and episode != 0:
            self.alpha = self.alpha / (episode / 1000)


def main(plot=True, env_name='CartPole-v0'):
    print("start training")
    ql = QLearning(env_name)

    # training
    ql()

    plt.plot(ql.rewards)
    plt.show()
    ql.test()
    ql.test()
    ql.test()


if __name__ == '__main__':
    main()
