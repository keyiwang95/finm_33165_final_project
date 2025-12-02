# networks_sac.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QNetworkContinuous(nn.Module):
    """
    Q(s, a) 网络：输入 [state, action] 拼接后的向量。
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
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    高斯策略：
    输入 state，输出均值、log_std，采样 z ~ N(mean, std)，
    经过 tanh 压缩到 [-1, 1]，再返回 action 和 log_prob。
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

        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _sample(self, mean: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        # reparameterization trick
        noise = torch.randn_like(mean)
        z = mean + std * noise
        action = torch.tanh(z)  # [-1, 1]

        # 计算 log_prob，补偿 tanh 的变换
        log_prob = (
            -0.5 * ((z - mean) ** 2 / (std ** 2 + 1e-6) + 2 * log_std + torch.log(torch.tensor(2 * torch.pi)))
        ).sum(dim=-1, keepdim=True)

        # tanh 变换的 log|det Jacobian|
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1, keepdim=True)
        return action, log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.base(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        action, log_prob = self._sample(mean, log_std)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        x = self.base(state)
        mean = self.mean_head(x)
        return torch.tanh(mean)

