from table.base import TableBase
import matplotlib.pyplot as plt


class Sarsa(TableBase):
    def __init__(self, env_name, num_episodes=5000, alpha=0.9, gamma=0.9, epsilon=1e-2):
        super(Sarsa, self).__init__(env_name, num_episodes, alpha, gamma, epsilon, policy="epsilon_greedy")

    def _loop(self):
        done = False
        total_reward, reward = 0, 0
        self.state = self.env.reset()
        action = self.policy()
        while not done:
            _state, reward, done, _ = self.env.step(action)
            _action = self.argmax(self.q_table[_state])
            self.q_table[self.state, action] += self.alpha * (
                reward + self.gamma * self.q_table[_state, _action] - self.q_table[self.state, action])
            total_reward += reward
            self.state = _state
            action = _action
        return total_reward


def main(plot=True, env_name="Taxi-v2", test_init_state=77):
    print("start training")
    sarsa9 = Sarsa(env_name, alpha=0.9)
    sarsa5 = Sarsa(env_name, alpha=0.5)
    sarsa1 = Sarsa(env_name, alpha=0.1)

    # training
    sarsa9()
    sarsa5()
    sarsa1()

    print("testing")
    print("gamma=0.9")
    sarsa9.test(test_init_state)
    print("gamma=0.5")
    sarsa5.test(test_init_state)
    print("gamma=0.1")
    sarsa1.test(test_init_state)

    if plot:
        plt.plot(sarsa1.rewards, label="alpha=0.1", alpha=0.5)
        plt.plot(sarsa5.rewards, label="alpha=0.5", alpha=0.5)
        plt.plot(sarsa9.rewards, label="alpha=0.9", alpha=0.5)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
