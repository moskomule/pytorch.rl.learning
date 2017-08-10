"""
original is from https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym
"""

from base import TableRLBase


class QLearing(TableRLBase):
    def __init__(self, env_name, num_episodes=10000, alpha=0.9, epsilon=1e-2):
        super(QLearing, self).__init__(env_name, num_episodes, alpha, epsilon)

    def _loop(self):
        done = False
        total_reward, reward = 0, 0
        self.state = self.env.reset()
        while not done:
            action = self.epsilon_greedy()

            _state, reward, done, _ = self.env.step(action)
            self.q_table[self.state, action] += self.alpha * (
                reward + self.q_table[_state].max() - self.q_table[self.state, action])
            total_reward += reward
            self.state = _state
        return total_reward


def main(plot=True, env_name="Taxi-v2", test_init_state=7):
    print("start training")
    ql = QLearing(env_name)
    # training
    ql()
    ql.test(test_init_state)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(ql.rewards)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
