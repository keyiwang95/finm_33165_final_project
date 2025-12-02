# run_sac_backtest.py

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .sac_agent import SACAgent
from .env_continuous import ContinuousPortfolioEnv


def run_backtest():

    # -----------------------------
    # 1. 加载数据
    # -----------------------------
    project_root = Path(__file__).resolve().parents[1]
    price_path = project_root / "price_data.pkl"

    price_df = pickle.load(open(price_path, "rb"))
    price_df = price_df.dropna()

    selected_cols = [
        ('Open', 'AAPL'),
        ('Open', 'AMZN'),
        ('Open', 'AMD'),
        ('Open', 'BAC'),
    ]
    price_df = price_df[selected_cols]

    n_assets = price_df.shape[1]

    # -----------------------------
    # 2. 创建环境
    # -----------------------------
    env = ContinuousPortfolioEnv(price_df=price_df, window=20)

    obs, _ = env.reset()

    # -----------------------------
    # 3. 初始化 agent 并加载模型参数
    # -----------------------------
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=n_assets,
    )

    model_path = project_root / "saved_models" / "sac_portfolio_model.pth"
    agent.load(model_path)

    # -----------------------------
    # 4. 回测（完全 deterministic，不加噪声）
    # -----------------------------
    done = False
    equity_curve = []
    weights_history = []

    while not done:
        action = agent.select_action(obs, eval_mode=True)  # 不加噪声
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        equity_curve.append(info["portfolio_value"])
        weights_history.append(env.weights.copy())

    # -----------------------------------------------------
    # 5) 计算 Sharpe / MaxDD / CAGR / Calmar，并保存到 metrics_sac.txt
    # -----------------------------------------------------
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak

    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)

    max_dd = drawdown.min()

    years = len(equity_curve) / 252  # 按 252 个交易日算 1 年
    CAGR = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1
    calmar = CAGR / abs(max_dd) if max_dd != 0 else np.inf

    project_root = Path(__file__).resolve().parents[2]   # FINAL_PROJECT/
    results_dir = project_root / "results_sac"
    results_dir.mkdir(exist_ok=True)

    metrics_path = results_dir / "metrics_sac.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Final Portfolio Value: {equity_curve[-1]:.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe:.4f}\n")
        f.write(f"Max Drawdown: {max_dd:.4f}\n")
        f.write(f"CAGR: {CAGR:.4f}\n")
        f.write(f"Calmar Ratio: {calmar:.4f}\n")

    print(f"SAC backtest metrics saved to {metrics_path}")

    # -----------------------------
    # 6. plot result
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

