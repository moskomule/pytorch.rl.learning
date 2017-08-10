from base import TableRLBase


class Sarsa(TableRLBase):
    def __init__(self, env_name, num_episodes=10000, alpha=0.9, epsilon=1e-2, gamma=0.9):
        super(Sarsa, self).__init__(env_name, num_episodes, alpha, epsilon)
        self.gamma = gamma

    def _loop(self):
        done = False
        total_reward, reward = 0, 0
        self.state = self.env.reset()
        action = self.epsilon_greedy()
        while not done:
            _state, reward, done, _ = self.env.step(action)
            _action = self.q_table[_state].max(dim=0)[1][0]
            self.q_table[self.state, action] += self.alpha * (
                reward + self.gamma * self.q_table[_state, _action] - self.q_table[self.state, action])
            total_reward += reward
            self.state = _state
            action = _action
        return total_reward


def main(plot=True, env_name="Taxi-v2", test_init_state=7):
    print("start training")
    sarsa9 = Sarsa(env_name, gamma=0.9)
    sarsa5 = Sarsa(env_name, gamma=0.5)
    sarsa1 = Sarsa(env_name, gamma=0.1)

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
        import matplotlib.pyplot as plt
        plt.plot(sarsa1.rewards, label="gamma=0.1", alpha=0.5)
        plt.plot(sarsa5.rewards, label="gamma=0.5", alpha=0.5)
        plt.plot(sarsa9.rewards, label="gamma=0.9", alpha=0.5)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main(plot=False)
