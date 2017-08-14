import gym
from torch import Tensor


class Base(object):
    """
    abstracted class for reinforcement learning scripts in `pre_dr`
    """

    def __init__(self, env_name, num_episodes, alpha, gamma, policy, report_freq=100, **kwargs):
        """
        base class for RL using lookup table
        :param env_name: name of environment, currently environments whose observation space and action space are
        both Discrete are supported. see https://github.com/openai/gym/wiki/Table-of-environments
        :param num_episodes: number of episode for training
        :param alpha:
        :param gamma:
        :param kwargs: other arguments.
        """
        self.env = gym.make(env_name)
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.state = None
        self._rewards = None
        self._policy = policy
        self.report_freq = report_freq
        for k, v in kwargs.items():
            setattr(self, str(k), v)

    def policy(self) -> int:
        """
        epsilon greedy method
        :return: action (int)
        """
        return getattr(self, self._policy)

    def schedule_alpha(self, episode: int):
        """
        schedule learning rate, this is optional
        :param episode: int
        """
        pass

    def _loop(self):
        """
        Loop in an episode. You need to implement.
        :return: total_reward (list)
        """
        raise NotImplementedError

    def train(self):
        """
        training the model
        """
        total_reward_list = []
        for episode in range(self.num_episodes):
            self.schedule_alpha(episode)
            total_reward = self._loop()
            total_reward_list.append(total_reward)

            if episode % self.report_freq == 0:
                print(f"episode:{episode} total reward:{total_reward:.2f}")
        self._rewards = total_reward_list

    def test(self, init_state=-1):
        """
        testing the trained model
        :param init_state: the initial state
        """
        raise NotImplementedError

    __call__ = train

    @property
    def rewards(self):
        return self._rewards

    @staticmethod
    def argmax(x):
        if isinstance(x, Tensor):
            return x.max(dim=0)[1][0]
        elif isinstance(x, list):
            return x.index(max(x))
