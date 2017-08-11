"""
inspired by https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym
"""

from table.base import TableRLBase


class QLearing(TableRLBase):
    def __init__(self, env_name, num_episodes=5000, alpha=0.9, gamma=0.9, epsilon=1e-2):
        super(QLearing, self).__init__(env_name, num_episodes, alpha, gamma, epsilon)

    def _loop(self):
        done = False
        total_reward, reward = 0, 0
        self.state = self.env.reset()
        while not done:
            action = self.epsilon_greedy()

            _state, reward, done, _ = self.env.step(action)
            self.q_table[self.state, action] += self.alpha * (
                reward + self.gamma * self.q_table[_state].max() - self.q_table[self.state, action])
            total_reward += reward
            self.state = _state
        return total_reward


def main(plot=True, env_name="Taxi-v2", test_init_state=77):
    print("start training")
    ql9 = QLearing(env_name, alpha=0.9)
    ql5 = QLearing(env_name, alpha=0.5)
    ql1 = QLearing(env_name, alpha=0.1)
    # training
    ql9()
    ql5()
    ql1()
    ql9.test(test_init_state)
    ql5.test(test_init_state)
    ql1.test(test_init_state)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(ql1.rewards, label="alpha=0.1", alpha=0.5)
        plt.plot(ql5.rewards, label="alpha=0.5", alpha=0.5)
        plt.plot(ql9.rewards, label="alpha=0.9", alpha=0.5)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
