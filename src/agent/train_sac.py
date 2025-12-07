# train_sac.py
"""
Train a Soft Actor–Critic (SAC) agent on portfolio allocation
using price data loaded from open_prices.parquet.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Import environment and SAC agent modules
from .env_continuous import ContinuousPortfolioEnv
from .sac_agent import SACAgent

# Import TICKERS from src/constants.py
# Directory structure:
#   src/constants.py
from src.constants import TICKERS


def save_plot(fn: str) -> None:
    """
    Save the current Matplotlib figure inside results_sac/plots/{fn}.png.
    Ensures the directory exists before saving.
    """
    plt.tight_layout()
    project_root = Path(__file__).resolve().parents[2]
    save_dir = project_root / "results_sac" / "new_plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{fn}.png", dpi=200)
    plt.close()


def train_sac(
    price_path: str = "open_prices.parquet",
    num_episodes: int = 50,
) -> None:
    """
    Train a SAC agent using a given price dataset (.parquet),
    then run a deterministic backtest and save performance plots.
    """

    # ---------------------------------------------------------
    # 1) Load price data from open_prices.parquet
    # ---------------------------------------------------------
    price_path_path = Path(price_path)
    assert price_path_path.exists(), f"{price_path_path} does not exist."

    # Load parquet file
    price_df: pd.DataFrame = pd.read_parquet(price_path_path)

    # Ensure the index is a DatetimeIndex
    if "date" in price_df.columns:
        price_df["date"] = pd.to_datetime(price_df["date"])
        price_df = price_df.set_index("date")
    else:
        price_df.index = pd.to_datetime(price_df.index)

    price_df = price_df.sort_index()
    price_df = price_df.dropna()

    # ---------------------------------------------------------
    # 2) Select only tickers defined in constants.TICKERS
    # ---------------------------------------------------------
    # Example: TICKERS = ["NVDA", "LLY", "JPM", "CAT"]
    selected_cols = [t for t in TICKERS if t in price_df.columns]

    assert (
        len(selected_cols) > 0
    ), "None of the tickers listed in TICKERS are present in the dataset."

    price_df = price_df[selected_cols]

    # ---------------------------------------------------------
    # 3) Restrict data to dates up to December 31, 2022
    # ---------------------------------------------------------
    cutoff = pd.Timestamp("2022-12-31")
    price_df = price_df.loc[price_df.index <= cutoff]

    # Remove any remaining missing values
    price_df = price_df.dropna()

    # Number of assets after filtering
    n_assets: int = price_df.shape[1]

    # ---------------------------------------------------------
    # 4) Initialize training environment
    # ---------------------------------------------------------
    # The environment handles:
    # - State construction (past window of prices)
    # - Action interpretation as portfolio weights
    # - Portfolio value evolution
    env: ContinuousPortfolioEnv = ContinuousPortfolioEnv(
        price_df=price_df,
        window=20,               # number of past days in the observation
        initial_cash=1_000_000.0 # starting capital
    )

    # Get dimension of the observation space
    sample_obs, _ = env.reset()
    state_dim: int = sample_obs.shape[0]
    action_dim: int = n_assets

    # ---------------------------------------------------------
    # 5) Create SAC agent (policy + value networks)
    # ---------------------------------------------------------
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        buffer_size=100_000,
        batch_size=256,
    )

    # Logging containers
    rewards_per_episode: List[float] = []
    q_loss_per_episode: List[float] = []
    policy_loss_per_episode: List[float] = []

    # ---------------------------------------------------------
    # 6) Training loop (episodes)
    # ---------------------------------------------------------
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done: bool = False
        ep_reward: float = 0.0
        ep_q_losses: List[float] = []
        ep_pi_losses: List[float] = []

        while not done:
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(
                state=obs,
                action=action,
                reward=float(reward),
                next_state=next_obs,
                done=float(done),
            )

            q_loss, pi_loss = agent.update()
            if q_loss is not None:
                ep_q_losses.append(float(q_loss))
            if pi_loss is not None:
                ep_pi_losses.append(float(pi_loss))

            obs = next_obs
            ep_reward += float(reward)

        rewards_per_episode.append(ep_reward)
        q_loss_per_episode.append(
            float(np.mean(ep_q_losses)) if ep_q_losses else float("nan")
        )
        policy_loss_per_episode.append(
            float(np.mean(ep_pi_losses)) if ep_pi_losses else float("nan")
        )

        print(
            f"[SAC] Episode {ep+1}/{num_episodes} | "
            f"Reward: {ep_reward:.2f} | "
            f"Q_loss: {q_loss_per_episode[-1]:.4f} | "
            f"Pi_loss: {policy_loss_per_episode[-1]:.4f}"
        )

    # ========================
    # Plot training curves
    # ========================
    plt.plot(rewards_per_episode)
    plt.title("SAC Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    save_plot("sac_episode_reward")

    plt.plot(q_loss_per_episode)
    plt.title("SAC Q Loss")
    plt.xlabel("Episode")
    plt.ylabel("Q Loss")
    save_plot("sac_q_loss")

    plt.plot(policy_loss_per_episode)
    plt.title("SAC Policy Loss")
    plt.xlabel("Episode")
    plt.ylabel("Policy Loss")
    save_plot("sac_policy_loss")

    # ========================
    # Final backtest (deterministic policy)
    # ========================
    obs, _ = env.reset()
    done = False
    equity_values: List[float] = []
    weights_history: List[NDArray[np.float32]] = []

    while not done:
        action = agent.select_action(obs, eval_mode=True)
        next_obs, _, terminated, truncated, info = env.step(action)
        equity_values.append(float(info["portfolio_value"]))
        weights_history.append(env.weights.copy())
        done = terminated or truncated
        obs = next_obs

    equity_curve: NDArray[np.float32] = np.array(
        equity_values, dtype=np.float32
    )
    weights_arr: NDArray[np.float32] = np.array(
        weights_history, dtype=np.float32
    )
    peak: NDArray[np.float32] = np.maximum.accumulate(equity_curve)
    drawdown_curve: NDArray[np.float32] = (equity_curve - peak) / peak

    # ========================
    # Final performance plots
    # ========================
    plt.plot(equity_curve)
    plt.title("SAC Final Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    save_plot("sac_equity_curve")

    for i in range(weights_arr.shape[1]):
        plt.plot(weights_arr[:, i], label=f"Asset {i}")
    plt.title("SAC Portfolio Weights Over Time")
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.legend()
    save_plot("sac_portfolio_weights")

    plt.plot(drawdown_curve)
    plt.title("SAC Drawdown Curve")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    save_plot("sac_drawdown_curve")

    print("SAC training & backtest finished.")

    # ========================
    # Save SAC model and weight history
    # ========================
    project_root = Path(__file__).resolve().parents[1]  # one level above src/
    save_dir = project_root / "saved_models"
    save_dir.mkdir(exist_ok=True)

    # Save SAC model parameters
    model_path = save_dir / "sac_portfolio_model2.pth"
    agent.save(str(model_path))

    # Save portfolio weight history
    weights_path = save_dir / "weights_history.npy"
    np.save(weights_path, weights_arr)

    print(f"Weights history saved to {weights_path}")


if __name__ == "__main__":
    train_sac()
