# env_continuous.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple, cast
from numpy.typing import NDArray


ObsType = NDArray[np.float32]
ActType = NDArray[np.float32]


class ContinuousPortfolioEnv(gym.Env[ObsType, ActType]):
    """
    Continuous-action multi-asset portfolio rebalancing environment.
    The agent outputs a continuous vector in R^N, which is later mapped to valid portfolio weights.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_df: pd.DataFrame,
        window: int = 20,
        initial_cash: float = 1_000_000.0,
    ) -> None:
        super().__init__()

        assert isinstance(price_df, pd.DataFrame)
        assert price_df.shape[1] >= 2

        # Clean price data
        self.price_df: pd.DataFrame = price_df.dropna().astype(float)
        self.assets = list(self.price_df.columns)
        self.n_assets: int = len(self.assets)

        self.window: int = window
        self.initial_cash: float = float(initial_cash)

        self.prices: NDArray[np.float32] = self.price_df.values.astype(
            np.float32
        )  # shape (T, N)
        self.returns: NDArray[np.float32] = (
            self.prices[1:] / self.prices[:-1] - 1.0
        )
        self.T: int = len(self.returns)

        # Same state dimension as before
        self.state_dim: int = (self.n_assets * 4) + 1

        # Continuous action space in R^N (later normalized to valid weights)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Observation is an unconstrained vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Initialize environment state
        self.t: int = 0
        self.portfolio_value: float = self.initial_cash
        self.weights: NDArray[np.float32] = np.zeros(
            self.n_assets, dtype=np.float32
        )
        self._last_obs: NDArray[np.float32] = np.zeros(
            self.state_dim, dtype=np.float32
        )

        self.reset()

    def _compute_state(self, t: int) -> NDArray[np.float32]:
        """
        Build the state vector using financial statistics over the past window:
          - last return for each asset
          - mean return over window
          - volatility of returns over window
          - current portfolio weights
          - normalized portfolio value
        """
        past_returns: NDArray[np.float32] = self.returns[
            t - self.window : t
        ]  # shape (window, N)
        mean_ret: NDArray[np.float32] = past_returns.mean(axis=0)
        vol_ret: NDArray[np.float32] = past_returns.std(axis=0) + 1e-8
        last_ret: NDArray[np.float32] = self.returns[t - 1]

        state = np.concatenate(
            [
                last_ret,  # N
                mean_ret,  # N
                vol_ret,  # N
                self.weights,  # N
                np.array(
                    [self.portfolio_value / self.initial_cash],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        # Help mypy: ensure this is seen as NDArray[np.float32]
        return cast(NDArray[np.float32], state.astype(np.float32))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Reset the environment to the initial state.
        """
        super().reset(seed=seed)

        self.t = self.window
        self.portfolio_value = self.initial_cash
        self.weights = np.zeros(self.n_assets, dtype=np.float32)

        obs: NDArray[np.float32] = self._compute_state(self.t)
        self._last_obs = obs
        return obs, {}

    def _action_to_weights(self, action: ActType) -> NDArray[np.float32]:
        """
        Map a continuous action vector into valid portfolio weights.
        Method:
            ReLU + normalization → ensures non-negative weights summing to 1.
        """
        # Avoid degenerate all-negative or all-zero vectors
        x = np.maximum(action, 0.0) + 1e-6
        w = x / x.sum()

        return cast(NDArray[np.float32], w.astype(np.float32))

    def step(
        self,
        action: ActType,
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, float]]:
        """
        Execute a single environment step:
          1. Convert action to weights
          2. Compute portfolio return
          3. Update portfolio value
          4. Build next observation
        """
        action_np = np.asarray(action, dtype=np.float32)
        assert self.action_space.contains(action_np)

        # Convert raw action into weights
        self.weights = self._action_to_weights(action_np)

        # Portfolio return for this step
        asset_rets: NDArray[np.float32] = self.returns[self.t]  # shape (N,)
        port_ret: float = float(np.dot(self.weights, asset_rets))

        # Update portfolio value
        prev_value: float = self.portfolio_value
        self.portfolio_value *= 1.0 + port_ret

        # Reward = change in portfolio value (could be changed to log-return)
        reward: float = self.portfolio_value - prev_value

        # Advance time index
        self.t += 1
        terminated: bool = self.t >= self.T
        truncated: bool = False

        if not terminated:
            obs: NDArray[np.float32] = self._compute_state(self.t)
        else:
            # At termination, keep the last observation
            obs = self._last_obs

        self._last_obs = obs

        info: Dict[str, float] = {
            "portfolio_value": float(self.portfolio_value)
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Print the current portfolio state.
        """
        print(
            f"t={self.t}, "
            f"portfolio_value={self.portfolio_value:.2f}, "
            f"weights={self.weights}"
        )
