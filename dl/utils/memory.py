from random import sample
from collections import deque, namedtuple


# replay memory
class Memory(object):
    def __init__(self, max_size=None):
        self._container = deque(maxlen=max_size)

    def __call__(self, val):
        self._container.append(val)

    def __repr__(self):
        return str(self._container)

    def sample(self, batch_size):
        return sample(self._container, batch_size)

    @property
    def is_empty(self):
        return len(self._container) == 0


###
# state transition tuple
###
Transition = namedtuple("Transition", ["state_before", "action", "reward", "state_after", "done"])
