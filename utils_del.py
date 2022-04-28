import random
import time
import numpy as np
from torch.nn import functional as F
import torch
import ray
from dotmap import DotMap
from copy import deepcopy
from collections import deque
from typing import List, Any, Tuple, TypeVar
from dataclasses import dataclass
import functools
import collections
from argparse import Namespace

@dataclass
class Transition:
    obs: dict
    act: int
    reward: float
    done: bool
    info: dict
    next_obs: dict


def convert_nested_dict(d: dict) -> Namespace:
    x = Namespace()
    _ = [setattr(x, k, convert_nested_dict(v)) if isinstance(v, dict) else setattr(x, k, v) for k, v in d.items()]
    return x


def run_stats(name, values):
    return {
        f"{name}_mean": float(values.mean()),
        f"{name}_std": float(values.std()),
        f"{name}_max": float(values.max()),
        f"{name}_min": float(values.min()),
    }


class BestBuffer:
    def __init__(self, buffer_size: int):
        self._buffer = [None for _ in range(buffer_size)]
        self._scores = np.full(buffer_size, -np.inf)
        self._cnt = 0
        self._buffer_argmin = 0

    def buffer(self) -> Tuple[np.ndarray, List[Any]]:
        return self._scores[:self._cnt], self._buffer[:self._cnt]

    def extend(self, priorities: List[float], values: List[Any]) -> None:
        for _score, _add_v in zip(priorities, values):
            if self._scores[self._buffer_argmin] < _score:
                self._scores[self._buffer_argmin] = _score
                self._buffer[self._buffer_argmin] = _add_v
                self._buffer_argmin = self._scores.argmin()
                self._cnt += 1

    def __len__(self):
        return self._cnt


def add_new_leaves(history: set, traj: List[Transition], env_class) -> bool:
    """ Return if to filter out or not"""
    _id = env_class.unique_id(traj[-1].next_obs)
    if _id in history:
        return False
    else:
        history.update([_id])
        return True


EnvClass = TypeVar('EnvClass')  # Temporary dummy class


class OfflineBufferWithBest:
    def __init__(self, full_history_size: int, best_history_size: int,
                 env: EnvClass, min_size_start_sample: int = 0, add_filter=add_new_leaves):
        self.full_history_size, self.best_history_size = full_history_size, best_history_size
        self.min_size_start_sample = min_size_start_sample
        self._seen_keys = set()
        self.full_history = deque(maxlen=full_history_size)
        self.best_history = BestBuffer(best_history_size)
        self.add_filter = functools.partial(add_filter, env_class=env)
        self.env_class = env

        # Should log here different metrics about new sampled trajectories
        log_window = 1000
        self.sampled_r = collections.deque(maxlen=log_window)
        self.new_sampled_r = collections.deque(maxlen=log_window)
        self.new_samples = collections.deque(maxlen=log_window)

    def add_trajs(self, trajs: List[List[Transition]]) -> dict:
        add_filter = [self.add_filter(self._seen_keys, v) for v in trajs]
        self.new_samples.extend(add_filter)
        self.sampled_r.extend([traj[-1].reward for traj in trajs])
        self.full_history.extend(trajs)
        # Add to best_history only new leaves
        _filtered_tr = [tr[-1] for do_add, tr in zip(add_filter, trajs) if do_add]
        self.new_sampled_r.extend([tr.reward for tr in _filtered_tr])
        self.best_history.extend([t.reward for t in _filtered_tr], _filtered_tr)
        return {"new_leaf": add_filter}

    def sample_batch(self, sample_batch_size: int) -> List[List[Transition]]:
        _, buffer = self.best_history.buffer()
        sample_leaves = random.choices(buffer, k=sample_batch_size)
        trajs = [self.env_class.backward_sample(x) for x in sample_leaves]
        return trajs

    def has_for_sampling(self) -> bool:
        return len(self.best_history) > self.min_size_start_sample

    @property
    def trajs(self) -> List[List[Transition]]:
        return list(self.full_history)

    def clear(self) -> None:
        self._seen_keys = set()
        self.full_history.clear()
        self.best_history = BestBuffer(self.best_history_size)

    def log(self) -> dict:
        return {
            "uniq_leaves_count": len(self._seen_keys),
            **run_stats("samples_r", np.array(self.sampled_r)),
            **run_stats("new_samples_r", np.array(self.new_sampled_r)),
            **run_stats("new_samples", np.array(self.new_samples)),
        }

