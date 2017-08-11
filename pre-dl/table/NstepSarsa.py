from torch import Tensor
from table.base import TableRLBase


class NstepSarsa(TableRLBase):
    def __init__(self, env_name, n_steps, num_episodes=5000, alpha=0.9, gamma=0.9, epsilon=1e-2):
        super(NstepSarsa, self).__init__(env_name, num_episodes, alpha, gamma, epsilon, n_steps=n_steps)

    def _loop(self):
        done = False
        total_reward, reward = 0, 0
        self.state = self.env.reset()
        action = self.epsilon_greedy()
        while not done:
            # n = 1 (same as SARSA)
            _state, reward, done, _ = self.env.step(action)
            _action = self.q_table[_state].max(dim=0)[1][0]
            reward_hist = [reward]
            __action, __state = _action, _state
            # n > 1
            for idx in range(self.n_steps - 1):
                __state, __reward, done, _ = self.env.step(__action)
                __action = self.q_table[__state].max(dim=0)[1][0]
                reward_hist.append(__reward)
                if done:
                    break  # go outside

            reward_hist_size = len(reward_hist)
            reward = Tensor(reward_hist) @ Tensor([pow(self.gamma, i) for i in range(reward_hist_size)])
            q = reward + pow(self.gamma, reward_hist_size) * self.q_table[__state, __action]
            self.q_table[self.state, action] += self.alpha * (q - self.q_table[self.state, action])

            total_reward += reward
            self.state = __state
            action = __action

        return total_reward


def main(plot=True, env_name="Taxi-v2", test_init_state=77):
    print("start training")
    n_sarsa2 = NstepSarsa(env_name, num_episodes=50000, n_steps=2, epsilon=0.1)
    n_sarsa3 = NstepSarsa(env_name, num_episodes=50000, n_steps=3, epsilon=0.1)
    n_sarsa4 = NstepSarsa(env_name, num_episodes=50000, n_steps=4, epsilon=0.1)

    # training
    n_sarsa2()
    n_sarsa3()
    n_sarsa4()

    print("testing")
    n_sarsa2.test(test_init_state)
    n_sarsa3.test(test_init_state)
    n_sarsa4.test(test_init_state)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(n_sarsa2.rewards, label="n_steps=2", alpha=0.5)
        plt.plot(n_sarsa3.rewards, label="n_steps=3", alpha=0.5)
        plt.plot(n_sarsa4.rewards, label="n_steps=4", alpha=0.5)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
