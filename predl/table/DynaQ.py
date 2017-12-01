from base import Memory
from predl import TableBase, Array2D


class DynaQ(TableBase):
    def __init__(self, env_name, num_episodes=100, alpha=0.9, gamma=0.9, epsilon=1e-2, model_loop=3,
                 min_alpha=0.01, decay_freq=100):
        """
        :param model_loop: number of times using model to update Q-value
        """
        super(DynaQ, self).__init__(env_name, num_episodes, alpha, gamma, epsilon, policy="epsilon_greedy",
                                    model_loop=model_loop, min_alpha=min_alpha, decay_freq=decay_freq)
        self.m_table = Array2D(self.obs_size, self.action_size)
        self._history = Memory()

    def _loop(self):
        done = False
        total_reward, reward = 0, 0
        self.state = self.env.reset()
        while not done:
            action = self.policy()
            self._history((self.state, action))
            _state, reward, done, _ = self.env.step(action)
            self.q_table[self.state, action] += self.alpha * (
                    reward + self.gamma * self.q_table[_state].max() - self.q_table[self.state, action])
            self.m_table[self.state, action] = (reward, _state)
            # use model to update Q
            for _ in range(self.model_loop):
                s, a = self._history.sample()
                r, _s = self.m_table[s, a]
                self.q_table[s, a] += self.alpha * (
                        r + self.gamma * self.q_table[_s].max() - self.q_table[s, a])
            total_reward += reward
            self.state = _state
        return total_reward

    def schedule_alpha(self, episode):
        if self.alpha > self.min_alpha and episode % self.decay_freq == 0 and episode != 0:
            self.alpha = self.alpha / (episode / self.decay_freq)


def main(plot=True, env_name="Taxi-v2", test_init_state=77):
    print("start training")
    dyna = DynaQ(env_name)
    # training
    dyna()
    dyna.test(test_init_state)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(dyna.rewards, alpha=0.5)
        plt.show()


if __name__ == '__main__':
    main()
