# replay_buffer_sac.py
import random
from collections import deque
from typing import Deque, Tuple, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray
import torch


class Transition(NamedTuple):
    """
    A single experience tuple used for SAC training.

    Each field corresponds to:
        - state:       observation before the action
        - action:      action taken by the agent
        - reward:      scalar reward received
        - next_state:  observation after the action
        - done:        whether the episode ended (1.0) or not (0.0)
    """
    state: NDArray[np.float32]
    action: NDArray[np.float32]
    reward: float
    next_state: NDArray[np.float32]
    done: float


class ReplayBufferContinuous:
    """
    A fixed-size replay buffer for storing transitions used by SAC.

    The buffer stores Transition objects and supports uniform random sampling.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: NDArray[np.float32],
        action: NDArray[np.float32],
        reward: float,
        next_state: NDArray[np.float32],
        done: float,
    ) -> None:
        """
        Store a new transition in the replay buffer.
        """
        self.buffer.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Uniformly sample a batch of transitions and convert them to torch tensors.

        Returns:
            states, actions, rewards, next_states, dones
        """
        # batch is a list of Transition objects
        batch: Sequence[Transition] = random.sample(self.buffer, batch_size)

        # Build numpy arrays explicitly; this is mypy- and numpy-stub-friendly
        states_np: NDArray[np.float32] = np.stack(
            [tr.state for tr in batch], axis=0
        )
        actions_np: NDArray[np.float32] = np.stack(
            [tr.action for tr in batch], axis=0
        )
        rewards_np: NDArray[np.float32] = np.asarray(
            [tr.reward for tr in batch], dtype=np.float32
        )
        next_states_np: NDArray[np.float32] = np.stack(
            [tr.next_state for tr in batch], axis=0
        )
        dones_np: NDArray[np.float32] = np.asarray(
            [tr.done for tr in batch], dtype=np.float32
        )

        states = torch.tensor(states_np, dtype=torch.float32)
        actions = torch.tensor(actions_np, dtype=torch.float32)
        rewards = torch.tensor(rewards_np, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states_np, dtype=torch.float32)
        dones = torch.tensor(dones_np, dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Returns the number of stored transitions.
        """
        return len(self.buffer)