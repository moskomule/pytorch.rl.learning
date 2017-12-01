from dl import make_atari
from dqn import Agent, Trainer, DQN


def main(env, gamma, epsilon, final_epsilon, final_exp_step,
         lr, memory_size, target_update_freq, gradient_update_freq, batch_size, replay_start,
         val_freq, log_freq_by_step, log_freq_by_ep, log_dir, weight_dir):
    train_env = make_atari(env + "NoFrameskip-v4")
    val_env = make_atari(env + "NoFrameskip-v4", noop=False)

    agent = Agent(train_env, DQN, gamma=gamma, epsilon=epsilon, final_epsilon=final_epsilon,
                  final_exp_step=final_exp_step)
    trainer = Trainer(agent, val_env, lr=lr, memory_size=memory_size, target_update_freq=target_update_freq,
                      gradient_update_freq=gradient_update_freq, batch_size=batch_size, replay_start=replay_start,
                      val_freq=val_freq, log_freq_by_step=log_freq_by_step, log_freq_by_ep=log_freq_by_ep,
                      log_dir=log_dir, weight_dir=weight_dir)
    trainer.train(args.steps)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pong")
    p.add_argument("--steps", type=int, default=50_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1)
    p.add_argument("--final_epsilon", type=float, default=0.05)
    p.add_argument("--final_exp_step", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--memory_size", type=int, default=1_000_000)
    p.add_argument("--target_update_freq", type=int, default=10_000)
    p.add_argument("--gradient_update_freq", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--replay_start", type=int, default=50_000)
    p.add_argument("--val_freq", type=int, default=250_000)
    p.add_argument("--log_freq_by_step", type=int, default=500)
    p.add_argument("--log_freq_by_ep", type=int, default=10)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--weight_dir", default=None)
    args = p.parse_args()

    main(**args)
