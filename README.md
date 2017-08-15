This repository is for learning *reinforcement learning* using PyTorch.

# Contents
## pre-dl

[pre-dl](./pre-dl) contains some algorithms without Deep Learning. Here, PyTorch is just numpy-alternative.
* lookup table
    + Q-Learning
    + SARSA
        - n-step SARSA
        - SARSA(λ)
* function approximation
    + Q-learning
    + SARSA
* policy gradient
    + REINFORCE
    + REINFORCE with baseline
    + actor-critic

[dl](./dl) contains some algorithms using DL.
* No Contents

# Dependency

* [gym](https://gym.openai.com/)

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

+ 牧野貴樹ほか.これからの強化学習.2016.
    * 基礎から応用まで扱っています
