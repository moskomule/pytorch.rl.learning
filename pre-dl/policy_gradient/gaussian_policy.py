from policy_gradient.pg_base import PGBase


class GradientPolicy(PGBase):
    def __init__(self, env_name, num_episodes, alpha, gamma):
        super(GradientPolicy, self).__init__(env_name, num_episodes, alpha, gamma, policy="gaussian_policy")

    def _loop(self):
        pass

    def policy(self):
        pass

    def _initialize_weight(self):
        pass

    def feature(self, state, action):
        pass
