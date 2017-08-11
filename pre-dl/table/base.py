import gym
import torch
import random


class TableRLBase(object):
    def __init__(self, env_name, num_episodes, alpha, gamma, epsilon, **kwargs):
        """
        base class for RL using lookup table
        :param env_name: name of environment
        :param num_episodes: number of episode for training
        :param alpha:
        :param epsilon:
        """
        self.env = gym.make(env_name)
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
        if random.random() > self.epsilon:
            action = self.q_table[self.state].max(dim=0)[1][0]
        else:
            action = random.randrange(0, self.action_size)
        return action

    def _loop(self):
        """
        loop in an episode
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
            action = self.q_table[state].max(dim=0)[1][0]
            print(action)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            counter += 1
        print(f"total reward {total_reward} in {counter} steps")

    def __call__(self):
        return self.train()

    @property
    def rewards(self):
        return self._rewards
