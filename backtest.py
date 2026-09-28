"""向量化回测引擎（从零实现，不依赖 Backtrader）。

面试考点：为什么自己写回测而不是调库？
答：为了完全掌控交易成本、调仓时点、持仓权重的计算逻辑，
确保没有前视偏差（look-ahead bias）和幸存者偏差。

回测流程：
1. 每 rebalance_days 个交易日，用因子打分选前 top_n 只股票
2. 等权持仓，调仓时扣除双边交易成本（佣金 + 滑点）
3. 调仓日之间权重保持不变（hold）
4. 输出日频组合净值序列
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    top_n: int = 5                # 每期持有股票数
    rebalance_days: int = 20      # 调仓间隔（交易日）
    cost_per_side: float = 0.001  # 单边成本 0.1%（佣金+滑点）
    initial_capital: float = 1_000_000.0
    cash_buffer: float = 0.05     # 现金缓冲比例


def run_backtest(
    score: pd.DataFrame,
    close: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict:
    """向量化回测主函数。

    Args:
        score:  因子综合打分，index=日期, columns=股票（越大越看好）
        close:  收盘价矩阵
        config: 回测参数

    Returns:
        dict: daily_returns / nav / turnover / weights / gross_returns / config
    """
    cfg = config or BacktestConfig()

    # 对齐日期与股票
    common_dates = score.index.intersection(close.index)
    common_stocks = score.columns.intersection(close.columns)
    score = score.loc[common_dates, common_stocks]
    close = close.loc[common_dates, common_stocks]

    daily_ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n_days = len(common_dates)
    n_stocks = len(common_stocks)
    stock_pos = {s: i for i, s in enumerate(common_stocks)}

    # --- 在调仓日生成目标权重，其余日 NaN，最后 ffill 持有 ---
    weights_arr = np.full((n_days, n_stocks), np.nan)
    costs = np.zeros(n_days)
    turnover_list: list[tuple[pd.Timestamp, float]] = []

    prev_target = np.zeros(n_stocks)
    rebal_indices = list(range(0, n_days, cfg.rebalance_days))

    for idx in rebal_indices:
        row = score.iloc[idx].dropna()
        if len(row) > 0:
            # 选分数最高的 top_n 只，等权 + 现金缓冲
            selected = row.nlargest(min(cfg.top_n, len(row))).index
            target = np.zeros(n_stocks)
            for s in selected:
                target[stock_pos[s]] = (1.0 - cfg.cash_buffer) / len(selected)
        else:
            target = prev_target  # 无可用信号，维持原仓

        weights_arr[idx] = target

        # 换手率 = 买入卖出量的一半（双边口径）
        turnover = np.abs(target - prev_target).sum() / 2
        costs[idx] = turnover * cfg.cost_per_side * 2  # 买卖各收一次
        turnover_list.append((common_dates[idx], turnover))

        prev_target = target

    # ffill：调仓日之间持仓不变；开头未建仓区间补 0
    weights = (
        pd.DataFrame(weights_arr, index=common_dates, columns=common_stocks)
        .ffill()
        .fillna(0.0)
    )

    # --- 组合收益 ---
    w = weights.to_numpy()
    r = daily_ret.to_numpy()
    gross = np.sum(w * r, axis=1)
    net = gross - costs

    nav = (1.0 + pd.Series(net, index=common_dates)).cumprod()

    return {
        "daily_returns": pd.Series(net, index=common_dates),
        "gross_returns": pd.Series(gross, index=common_dates),
        "nav": nav,
        "turnover": pd.Series(
            [t for _, t in turnover_list],
            index=[d for d, _ in turnover_list],
            name="turnover",
        ),
        "weights": weights,
        "config": cfg,
    }


def buy_and_hold(close: pd.DataFrame) -> pd.Series:
    """基准：等权买入持有所有股票（剔除无效价格）。"""
    valid = close.where(close > 0).dropna(how="all")
    ret = valid.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + ret.mean(axis=1)).cumprod()
