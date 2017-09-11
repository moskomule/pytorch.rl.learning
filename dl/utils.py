from random import sample
from collections import deque

import numpy as np
import gym
from gym import spaces


########################
# replay memory
########################
class Memory(object):
    def __init__(self, max_size=None):
        self._container = ()
        self.max_size = max_size

    def __call__(self, val):
        if self.max_size is not None and len(self._container) == self.max_size:
            self._container = self._container[1:]
        self._container += (val,)

    def __repr__(self):
        return str(self._container)

    def sample(self, batch_size):
        return sample(self._container, batch_size)

    @property
    def is_empty(self):
        return len(self._container) == 0


########################
# gym's wrappers
# see https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers_deprecated.py
########################
class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env, frame_size=84):
        """
        reshape frames to square of given frame_size by frame_size
        """
        super(ProcessFrame, self).__init__(env)
        self.frame_size = frame_size
        self.observation_space = spaces.Box(low=0, high=255, shape=(frame_size, frame_size, 1))

    def _observation(self, obs):
        return self.process(obs)

    def process(self, frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        # to make input gray scale
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = np.resize(img, (110, self.frame_size))
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [self.frame_size, self.frame_size, 1])
        return x_t.astype(np.float32)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        remove flickering and return only every `skip`-th frame and
        """
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def _step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # This was necessary to remove flickering that is present in games where some objects appear only in even frames
        # while other objects appear only in odd frames
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)

        return obs


class LazyFrames(object):
    def __init__(self, frames, transpose=True):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was.
        """
        self._frames = frames
        self._transpose = transpose

    def __array__(self, dtype=None):
        # to treat LazyFrames object as NdArray
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        if self._transpose:
            out = out.transpose((2, 0, 1))
        return out


class ClippedRewardsWrapper(gym.RewardWrapper):
    def _reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    def __init__(self, env, stack_size, transpose=True):
        """
        Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        """
        super(FrameStack, self).__init__(env)
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)
        shape = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shape[0], shape[1], shape[2] * stack_size))
        self._transpose = transpose

    def _reset(self):
        ob = self.env.reset()
        for _ in range(self.stack_size):
            self.frames.append(ob)
        return self._get_ob()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.stack_size, "len(self.frames) != self.stack_size"
        return LazyFrames(list(self.frames), transpose=self._transpose)


def convert_env(env):
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrame(env)
    env = FrameStack(env, 4)
    env = ClippedRewardsWrapper(env)
    return env
