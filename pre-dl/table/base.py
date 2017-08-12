import gym
import torch
import random


class TableRLBase(object):
    def __init__(self, env_name, num_episodes, alpha, gamma, epsilon, **kwargs):
        """
        base class for RL using lookup table
        :param env_name: name of environment, currently environments whose observation space and action space are
        both Discrete are supported. see https://github.com/openai/gym/wiki/Table-of-environments
        :param num_episodes: number of episode for training
        :param alpha:
        :param gamma:
        :param epsilon:
        :param kwargs: other arguments.
        """
        self.env = gym.make(env_name)

        if not isinstance(self.env.action_space, gym.spaces.Discrete) or \
                not isinstance(self.env.observation_space, gym.spaces.Discrete):
            raise NotImplementedError("action_space and observation_space should be Discrete")

        self.obs_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.q_table = torch.zeros(self.obs_size, self.action_size)
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state = None
        self._rewards = None
        for k, v in kwargs.items():
            setattr(self, str(k), v)

    def epsilon_greedy(self):
        """
        epsilon greedy method
        :return: action (int)
        """
        _epsilon = self.epsilon * (1 - 1 / self.action_size)
        if random.random() > _epsilon:
            action = self.argmax(self.q_table[self.state])
        else:
            action = random.randrange(0, self.action_size)
        return action

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
            total_reward = self._loop()
            total_reward_list.append(total_reward)

            if episode % 100 == 0:
                print(f"episode:{episode} total reward:{total_reward:.2f}")
        self._rewards = total_reward_list

    def test(self, init_state=-1):
        """
        testing the trained model
        :param init_state: the initial state
        """
        done = False
        total_reward, reward, counter = 0, 0, 0
        state = self.env.reset() if init_state is -1 else init_state
        while not done:
            self.env.render()
            action = self.argmax(self.q_table[state])
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            counter += 1
        print(f"total reward {total_reward} in {counter} steps")

    __call__ = train

    @property
    def rewards(self):
        return self._rewards

    @staticmethod
    def argmax(x: torch.Tensor):
        return x.max(dim=0)[1][0]
