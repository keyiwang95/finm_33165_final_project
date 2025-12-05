# run_sac_backtest.py

import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .env_continuous import ContinuousPortfolioEnv
from .sac_agent import SACAgent


def run_backtest() -> None:
    """
    Run a deterministic backtest using a trained SAC model and
    compute basic performance metrics (Sharpe, MaxDD, CAGR, Calmar).
    """

    # -----------------------------
    # 1. Load price data
    # -----------------------------
    project_root = Path(__file__).resolve().parents[1]
    price_path = project_root / "price_data.pkl"

    price_df = pickle.load(open(price_path, "rb"))
    price_df = price_df.dropna()

    # Use selected assets
    selected_cols = [
        ("Open", "AAPL"),
        ("Open", "ABT"),
        ("Open", "MU"),
        ("Open", "SO"),
    ]
    price_df = price_df[selected_cols]

    n_assets = price_df.shape[1]

    # -----------------------------
    # 2. Create environment
    # -----------------------------
    env = ContinuousPortfolioEnv(price_df=price_df, window=20)

    obs, _ = env.reset()

    # -----------------------------
    # 3. Initialize SAC agent & load model weights
    # -----------------------------
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=n_assets,
    )

    model_path = project_root / "saved_models" / "sac_portfolio_model.pth"
    agent.load(str(model_path))  # Path -> str

    # -----------------------------
    # 4. Deterministic evaluation (no exploration noise)
    # -----------------------------
    done = False
    equity_values: List[float] = []
    weights_history: List[NDArray[np.float32]] = []

    while not done:
        # Deterministic action from the policy
        action = agent.select_action(obs, eval_mode=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        equity_values.append(float(info["portfolio_value"]))
        weights_history.append(env.weights.copy())

    # -----------------------------------------------------
    # 5. Compute Sharpe / MaxDD / CAGR / Calmar and save to metrics_sac.txt
    # -----------------------------------------------------
    equity_curve: NDArray[np.float64] = np.array(equity_values, dtype=np.float64)

    peak: NDArray[np.float64] = np.maximum.accumulate(equity_curve)
    drawdown: NDArray[np.float64] = (equity_curve - peak) / peak

    returns: NDArray[np.float64] = np.diff(equity_curve) / equity_curve[:-1]
    sharpe: float = float(np.mean(returns) / (np.std(returns) + 1e-8))

    max_dd: float = float(drawdown.min())

    years: float = len(equity_curve) / 252.0  # assume 252 trading days per year
    CAGR: float = float((equity_curve[-1] / equity_curve[0]) ** (1.0 / years) - 1.0)
    calmar: float = CAGR / abs(max_dd) if max_dd != 0.0 else float("inf")

    project_root_final = Path(__file__).resolve().parents[2]  # FINAL_PROJECT/
    results_dir = project_root_final / "results_sac"
    results_dir.mkdir(exist_ok=True)

    metrics_path = results_dir / "metrics_sac.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Final Portfolio Value: {equity_curve[-1]:.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe:.4f}\n")
        f.write(f"Max Drawdown: {max_dd:.4f}\n")
        f.write(f"CAGR: {CAGR:.4f}\n")
        f.write(f"Calmar Ratio: {calmar:.4f}\n")

    print(f"SAC backtest metrics saved to {metrics_path}")

    # -----------------------------------------------------
    # 6. Save weights/positions history to CSV & Excel
    # -----------------------------------------------------
    weights_arr = np.array(weights_history)  # shape (T, n_assets)
    df_weights = pd.DataFrame(
        weights_arr,
        columns=[str(col) for col in env.assets]
    )
    df_weights["PortfolioValue"] = equity_curve

    weights_csv_path = results_dir / "sac_weights_history.csv"
    weights_excel_path = results_dir / "sac_weights_history.xlsx"

    df_weights.to_csv(weights_csv_path, index=False)
    df_weights.to_excel(weights_excel_path, index=False)

    print(f"Weights history saved to:\n  {weights_csv_path}\n  {weights_excel_path}")


    # -----------------------------
    # 7. Plot results
    # -----------------------------
    plt.plot(equity_curve)
    plt.title("Equity Curve (Loaded SAC Model)")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.show()

    plt.plot(drawdown)
    plt.title("Drawdown Curve (Loaded SAC Model)")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.show()


if __name__ == "__main__":
    run_backtest()
