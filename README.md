This repository is for learning *reinforcement learning* using PyTorch.
# how to use

Run config file as below to set `PYTHONPATH`.

```
cd pre-dl # or dl
. ./config.sh
```

# Contents
## pre-dl

[pre-dl](./pre-dl) contains some algorithms without Deep Learning. Here, PyTorch is just numpy-alternative.
* [lookup table](./pre-dl/table)
    + Q-Learning
    + SARSA
        - n-step SARSA
        - SARSA(λ)
    + DynaQ (Model Free Approach)

* [function approximation](./pre-dl/function_approximation)
    + Q-learning
    + SARSA
* [policy gradient](./pre-dl/policy_gradient)
    + REINFORCE
    + REINFORCE with baseline
    + actor-critic

For simplicity, each algorithms inherit `FooBase` in `pre-dl/foo/foo_base.py` which inherits `RLBase` in `pre-dl/base.py`.

[dl](./dl) contains some algorithms using DL.
* [Deep Q-Network](./dl/dqn)

# Dependency

* [gym](https://gym.openai.com/)
    + `gym` requires a lot depends on your environment thus read the official document before using pip.

```
git clone https://github.com/openai/gym
cd gym
pip install -e . # minimal install
```

* [PyTorch](http://pytorch.org/)
* matplotlib

# References

+ [UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) by David Silver, DeepMind.
    * I mainly learned RL in this lecture.
    * Lecture videos are available on [YouTube](https://www.youtube.com/watch?v=2pWv7GOvuf0)

+ Richard S. Sutton and Andrew G. Barto Reinforcement Learning: An Introduction. 2017.
    * online draft is [available](http://incompleteideas.net/sutton/book/the-book-2nd.html)

+ [Introduction to reinforcement learning and OpenAI Gym](https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym) by Justin Francis
    * Good introduction to OpenAI Gym

+ 牧野貴樹ほか.これからの強化学習.2016.
    * 基礎から応用まで扱っています
