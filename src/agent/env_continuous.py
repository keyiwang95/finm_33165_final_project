# env_continuous.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class ContinuousPortfolioEnv(gym.Env):
    """
    连续动作版多资产组合再平衡环境：
    - 动作为 R^N 连续向量，经归一化后作为权重。
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_df: pd.DataFrame,
        window: int = 20,
        initial_cash: float = 1_000_000.0,
    ):
        super().__init__()

        assert isinstance(price_df, pd.DataFrame)
        assert price_df.shape[1] >= 2

        self.price_df = price_df.dropna().astype(float)
        self.assets = list(self.price_df.columns)
        self.n_assets = len(self.assets)

        self.window = window
        self.initial_cash = float(initial_cash)

        self.prices = self.price_df.values  # (T, N)
        self.returns = self.prices[1:] / self.prices[:-1] - 1.0
        self.T = len(self.returns)

        # state_dim 跟原来一致
        self.state_dim = (self.n_assets * 4) + 1

        # ✅ 这里改成连续动作空间：每个 asset 一个动作分量
        # 暂定范围 [-1, 1]，稍后通过 softmax / ReLU 转成合法权重
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.reset()

    def _compute_state(self, t: int) -> np.ndarray:
        past_returns = self.returns[t - self.window : t]  # (window, N)
        mean_ret = past_returns.mean(axis=0)
        vol_ret = past_returns.std(axis=0) + 1e-8
        last_ret = self.returns[t - 1]

        state = np.concatenate(
            [
                last_ret,               # N
                mean_ret,               # N
                vol_ret,                # N
                self.weights,           # N
                np.array([self.portfolio_value / self.initial_cash]),
            ],
            axis=0,
        )
        return state.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.window
        self.portfolio_value = self.initial_cash
        self.weights = np.zeros(self.n_assets, dtype=np.float32)

        obs = self._compute_state(self.t)
        self._last_obs = obs
        return obs, {}

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """
        将连续动作向量映射为合法权重：
        方案一：ReLU + 归一化，允许“接近0”的仓位
        """
        # 防止全负或全零
        x = np.maximum(action, 0.0) + 1e-6
        w = x / x.sum()
        return w.astype(np.float32)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action.astype(np.float32))

        # ✅ 将连续动作转权重
        self.weights = self._action_to_weights(action)

        asset_rets = self.returns[self.t]      # shape (N,)
        port_ret = np.dot(self.weights, asset_rets)
        prev_value = self.portfolio_value
        self.portfolio_value *= (1.0 + port_ret)

        reward = self.portfolio_value - prev_value

        self.t += 1
        terminated = self.t >= self.T
        truncated = False

        if not terminated:
            obs = self._compute_state(self.t)
        else:
            obs = self._last_obs

        self._last_obs = obs
        return obs, reward, terminated, truncated, {"portfolio_value": float(self.portfolio_value)}

    def render(self):
        print(
            f"t={self.t}, "
            f"portfolio_value={self.portfolio_value:.2f}, "
            f"weights={self.weights}"
        )
