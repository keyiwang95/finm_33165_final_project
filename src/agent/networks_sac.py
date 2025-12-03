# networks_sac.py
import torch
import torch.nn as nn
from typing import Tuple


class QNetworkContinuous(nn.Module):
    """
    Q(s, a) network for continuous actions.
    Input = concatenated [state, action] vector.
    Output = scalar Q-value.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: concatenate state and action along the last dimension.
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for SAC.

    Given a state:
      - Predict mean and log_std
      - Sample z ~ N(mean, std) with reparameterization trick
      - Apply tanh to squash to [-1, 1]
      - Return (action, log_prob) with tanh correction

    This is the standard SAC continuous policy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared MLP trunk
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Output heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _sample(self, mean: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action using:
            z = mean + std * noise
            action = tanh(z)

        Also compute the corrected log_prob after tanh transformation.
        """
        std = log_std.exp()

        # Reparameterization trick: noise is independent of weights
        noise = torch.randn_like(mean)
        z = mean + std * noise

        # Squash to [-1, 1]
        action = torch.tanh(z)

        # Log probability of Gaussian before tanh
        log_prob = (
            -0.5 * ((z - mean) ** 2 / (std ** 2 + 1e-6) + 2 * log_std + torch.log(torch.tensor(2 * torch.pi)))
        ).sum(dim=-1, keepdim=True)

        # Subtract tanh Jacobian correction
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1, keepdim=True)

        return action, log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass during training: sample stochastically.
        Returns:
            action ∈ [-1, 1]^N
            log_prob of the sampled action
        """
        x = self.base(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        action, log_prob = self._sample(mean, log_std)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic policy for evaluation:
        action = tanh(mean(state))
        """
        x = self.base(state)
        mean = self.mean_head(x)
        return torch.tanh(mean)


