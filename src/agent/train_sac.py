# train_sac.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from .env_continuous import ContinuousPortfolioEnv
from .sac_agent import SACAgent


def save_plot(fn):
    plt.tight_layout()
    project_root = Path(__file__).resolve().parents[2]
    save_dir = project_root / "results_sac/plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{fn}.png", dpi=200)
    plt.close()


def train_sac(
    price_path: str = "price_data.pkl",
    num_episodes: int = 50,
):
    price_path = Path(price_path)
    assert price_path.exists(), f"{price_path} does not exist."
    price_df: pd.DataFrame = pickle.load(open(price_path, "rb"))
    price_df = price_df.dropna()

    selected_cols = [
        ('Open', 'AAPL'),
        ('Open', 'AMZN'),
        ('Open', 'AMD'),
        ('Open', 'BAC'),
    ]
    price_df = price_df[selected_cols]

    n_assets = price_df.shape[1]

    env = ContinuousPortfolioEnv(
        price_df=price_df,
        window=20,
        initial_cash=1_000_000.0,
    )

    sample_obs, _ = env.reset()
    state_dim = sample_obs.shape[0]
    action_dim = n_assets

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

    rewards_per_episode = []
    q_loss_per_episode = []
    policy_loss_per_episode = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_q_losses = []
        ep_pi_losses = []

        while not done:
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(obs, action, reward, next_obs, float(done))
            q_loss, pi_loss = agent.update()
            if q_loss is not None:
                ep_q_losses.append(q_loss)
            if pi_loss is not None:
                ep_pi_losses.append(pi_loss)

            obs = next_obs
            ep_reward += reward

        rewards_per_episode.append(ep_reward)
        q_loss_per_episode.append(np.mean(ep_q_losses) if ep_q_losses else np.nan)
        policy_loss_per_episode.append(np.mean(ep_pi_losses) if ep_pi_losses else np.nan)

        print(
            f"[SAC] Episode {ep+1}/{num_episodes} | "
            f"Reward: {ep_reward:.2f} | "
            f"Q_loss: {q_loss_per_episode[-1]:.4f} | "
            f"Pi_loss: {policy_loss_per_episode[-1]:.4f}"
        )

    # 画训练曲线
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

    # 最终回测（贪心：使用 deterministic 策略）
    obs, _ = env.reset()
    done = False
    equity_curve = []
    weights_history = []

    while not done:
        action = agent.select_action(obs, eval_mode=True)
        next_obs, _, terminated, truncated, info = env.step(action)
        equity_curve.append(info["portfolio_value"])
        weights_history.append(env.weights.copy())
        done = terminated or truncated
        obs = next_obs

    equity_curve = np.array(equity_curve)
    weights_arr = np.array(weights_history)
    peak = np.maximum.accumulate(equity_curve)
    drawdown_curve = (equity_curve - peak) / peak

    # 曲线图（和 DDQN 基本一样）
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
    project_root = Path(__file__).resolve().parents[1]  # src/ 上一级（FINAL_PROJECT）
    save_dir = project_root / "saved_models"
    save_dir.mkdir(exist_ok=True)

    # 保存 SAC 模型参数
    model_path = save_dir / "sac_portfolio_model2.pth"
    agent.save(model_path)
    
    # 保存权重变动记录 (weights_history)
    # ============================
    weights_history = np.array(weights_history)  # list -> numpy array
    weights_path = save_dir / "weights_history.npy"
    np.save(weights_path, weights_history)

    print(f"Weights history saved to {weights_path}")

    


if __name__ == "__main__":
    train_sac()
