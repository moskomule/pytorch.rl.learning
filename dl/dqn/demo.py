from time import sleep
import torch

from dqn import DQN, variable, random, to_tensor, randrange
from dl.utils.wrapper import make_atari


def main(env, weight_path, epsilon):
    env = make_atari(env)
    q_function = DQN(env.action_space.n)
    q_function.load_state_dict(torch.load(weight_path))

    done = False
    state = env.reset()
    step = 1
    sleep(2)
    while not done:
        env.render()
        if random() <= epsilon:
            action = randrange(0, env.action_space.n)
        else:
            state = variable(to_tensor(state).unsqueeze(0))
            action = q_function(state).data.view(-1).max(dim=0)[1].sum()

        state, reward, done, info = env.step(action)
        print(f"[step: {step:>5}] [reward: {reward:>5}]")
        step += 1
    sleep(2)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("env")
    p.add_argument("weight_path")
    p.add_argument("--epsilon", type=float, default=0.05)
    args = p.parse_args()
    main(**vars(args))
