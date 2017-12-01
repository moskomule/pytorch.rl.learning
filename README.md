This repository is for learning *reinforcement learning* using PyTorch.
# how to use

Set this root dir as `PYTHONPATH`.

```
export PYTHONPATH=$(pwd)
```

# Contents
## predl

[predl](./predl) contains some algorithms without Deep Learning. Here, PyTorch is just numpy-alternative.
* [lookup table](./predl/table)
    + Q-Learning
    + SARSA
        - n-step SARSA
        - SARSA(λ)
    + DynaQ (Model Free Approach)

* [function approximation](./predl/function_approximation)
    + Q-learning
    + SARSA
* [policy gradient](./predl/policy_gradient)
    + REINFORCE
    + REINFORCE with baseline
    + actor-critic

## dl

[dl](./dl) contains some algorithms using DL.
* [Deep Q-Network](./dl/dqn)
    ```
    cd dl/dqn
    python exec.py [--env Pong ...] # `python exec.py -h` for help
    ---
    tensorboard log_dir runs
    ```

# Dependencies

* [gym](https://gym.openai.com/)
    + `gym` requires a lot depends on your environment thus read the official document before using pip.
* [PyTorch](http://pytorch.org/)
* matplotlib
* tensorboard-pytorch
    + `pip install tensorboardX`


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
