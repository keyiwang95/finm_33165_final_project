"""Trading enviroment."""

import gymnasium
import numpy as np


class TradingEnv(gymnasium.Env):
    """Trading enviroment."""

    def __init__(
        self,
        prices: np.ndarray,
        calc_window: int = 20,
        initial_cash: float = 10_000.0,
    ):
        super().__init__()

        self._prices = prices
        self._calc_window = calc_window
        self._initial_cash = initial_cash
