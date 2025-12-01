# -*- coding: utf-8 -*-

# Evaluation for Portfolio Rebalancing Model
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Expected data: T rows (T = length of time series); columns: data, portfolio value, reward, N x weights, N x rewards (N = number of assets)

# generating some simulated example data to test the evaluation functions:
T = 100
assets = ["A","B","C"]

df = pd.DataFrame({
    "portfolio_value": 1000*np.cumprod(1+0.001*np.random.randn(T)),
    "reward": np.random.randn(T)*0.1,
    "action": np.random.randint(0,4,size=T),
})
w = np.random.rand(T, len(assets))
w /= w.sum(axis=1, keepdims=True)

for i,a in enumerate(assets):
    df[f"weights_{a}"] = w[:,i]

#-------------------------------------------------------------------------------------

# Evaluation Plot (1): stacked area chart (probably only for few assets but is a nice visualization)

def plot_weights_area(df: pd.DataFrame, assets: List[str]) -> None: 
    df[[f"weights_{a}" for a in assets]].plot.area(figsize=(10,4))
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.title("Portfolio Weights Over Time")
    plt.show()

assets = ["A","B","C"]
plot_weights_area(df, assets)

# Evaluation Plot (2): cumulative return

def plot_cum_return(df: pd.DataFrame) -> None:
    ((1 + df["reward"]).cumprod() - 1).plot(figsize=(10,4))
    plt.title("Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.show()

# call on example data
plot_cum_return(df)

# Evaluation Plot (3): drawdown curve

def plot_drawdown(df: pd.DataFrame) -> None:
    eq = df["portfolio_value"]
    dd = eq - eq.cummax()
    dd.plot(figsize=(10,4))
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.show()

plot_drawdown(df)

# Evaluation Plot (4): rolling Sharpe-ratio
def plot_rolling_sharpe(df: pd.DataFrame, window: int = 30) -> None:
    r = df["reward"]
    sharpe = r.rolling(window).mean() / r.rolling(window).std()
    sharpe.plot(figsize=(10,4))
    plt.title("Rolling Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.show()

plot_rolling_sharpe(df)
