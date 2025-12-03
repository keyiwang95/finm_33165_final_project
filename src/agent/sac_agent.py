# sac_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple

from .networks_sac import QNetworkContinuous, GaussianPolicy
from .replay_buffer_sac import ReplayBufferContinuous


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        device: Optional[str] = None,
    ):
        """
        Soft Actor-Critic agent for continuous action spaces.
        Includes:
            - Gaussian policy (actor)
            - Two Q-networks + target networks (critics)
            - Replay buffer
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        # Policy network (actor)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)

        # Two Q-networks (SAC double critic)
        self.q1 = QNetworkContinuous(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetworkContinuous(state_dim, action_dim, hidden_dim).to(self.device)

        # Target Q-networks
        self.q1_target = QNetworkContinuous(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetworkContinuous(state_dim, action_dim, hidden_dim).to(self.device)

        # Initialize target networks with same parameters
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = ReplayBufferContinuous(buffer_size)

    # ----------- Action selection ----------
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Choose action given current state.
        eval_mode = True → deterministic policy (for evaluation/backtesting)
        eval_mode = False → stochastic policy (for exploration during training)
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if eval_mode:
                action = self.policy.deterministic(state_t)
            else:
                action, _ = self.policy(state_t)

        return action.cpu().numpy()[0]

    # ----------- SAC update step ----------
    def update(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Perform one SAC update using a minibatch sampled from the replay buffer.
        Returns:
            q_loss, policy_loss  (for logging)
        """

        # Not enough samples yet
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ------- 1) Update Q networks -------
        with torch.no_grad():
            # Next-state actions and log-probs
            next_actions, next_log_prob = self.policy(next_states)

            # Compute target Q using target networks
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next_min = torch.min(q1_next, q2_next)

            # Soft Bellman backup
            target_q = rewards + (1.0 - dones) * self.gamma * (q_next_min - self.alpha * next_log_prob)

        # Current Q estimates
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)

        # MSE losses
        q1_loss = nn.MSELoss()(q1_pred, target_q)
        q2_loss = nn.MSELoss()(q2_pred, target_q)
        q_loss = q1_loss + q2_loss

        # Gradient update for critics
        self.q1_optim.zero_grad()
        self.q2_optim.zero_grad()
        q_loss.backward()
        self.q1_optim.step()
        self.q2_optim.step()

        # ------- 2) Update policy -------
        # Freeze critic parameters
        for p in self.q1.parameters():
            p.requires_grad = False
        for p in self.q2.parameters():
            p.requires_grad = False

        # New candidate actions
        new_actions, log_prob = self.policy(states)

        # Evaluate actions under critics
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new_min = torch.min(q1_new, q2_new)

        # SAC policy objective
        policy_loss = (self.alpha * log_prob - q_new_min).mean()

        # Update actor
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Unfreeze critics
        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True

        # ------- 3) Soft update of target networks -------
        with torch.no_grad():
            for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

        # Return Q-loss and policy-loss for logging
        return float(q_loss.item()), float(policy_loss.item())

    # ----------- Save model -----------
    def save(self, path: str):
        """
        Save policy and critic parameters to disk.
        """
        ckpt = {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
        }
        torch.save(ckpt, path)
        print(f"[SAC] Model saved to {path}")

    # ----------- Load model -----------
    def load(self, path: str):
        """
        Load previously saved SAC model parameters.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        print(f"[SAC] Model loaded from {path}")

