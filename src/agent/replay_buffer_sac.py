# replay_buffer_sac.py
import random
from collections import deque, namedtuple
from typing import Deque, Tuple

import numpy as np
import torch

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done"),
)


class ReplayBufferContinuous:
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))

        states = torch.tensor(np.stack(batch.state), dtype=torch.float32)
        actions = torch.tensor(np.stack(batch.action), dtype=torch.float32)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

