"""因子计算模块。

因子（Factor）= 能够解释/预测未来收益的变量。
本模块实现三个经典因子：

1. 动量因子 Momentum   —— 过去20日收益率的截面排名（强者恒强）
2. 低波动因子 LowVol   —— 过去20日收益率标准差的截面排名（越低越好）
3. 短期反转因子 Reversal—— 过去5日收益率的截面排名（超涨回调）

所有因子统一输出"越大越好"的方向（正向化），
方便后续加权合成综合打分。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面标准化：每天在所有股票之间做百分位排名，映射到 [0, 1]。

    为什么用 rank 而不是 z-score？
    - rank 对极端值稳健（金融数据重尾，z-score 会被离群值带偏）
    - rank 输出天然有界，多因子加权时量纲统一
    """
    return df.rank(axis=1, pct=True)


def momentum_factor(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """动量因子：过去 window 日累计收益率的截面排名。

    经济学含义：资金持续流入的股票短期内有惯性。
    """
    ret = close.pct_change(window)
    return _cross_sectional_rank(ret)


def volatility_factor(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """低波动因子：过去 window 日收益率波动率的截面排名取反。

    经济学含义：低波动异象——高波动股票风险调整后收益反而更差
    （Baker & Haugen, 2012）。这里取排名的 (1 - rank) 实现"越低越好"。
    """
    vol = close.pct_change().rolling(window).std()
    rank = _cross_sectional_rank(vol)
    return 1.0 - rank


def reversal_factor(close: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """短期反转因子：过去 window 日收益率的截面排名取反。

    经济学含义：短期超涨的股票倾向于回调（De Bondt & Thaler, 1985）。
    注意与动量因子窗口不同——动量看中期，反转看短期，二者可共存。
    """
    ret = close.pct_change(window)
    rank = _cross_sectional_rank(ret)
    return 1.0 - rank


def composite_score(
    close: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """多因子合成综合打分。

    score = w1 * momentum + w2 * lowvol + w3 * reversal

    权重设计思路（面试可讲）：
    - 初始等权是最诚实的起点——没有充分证据前不应过拟合权重
    - 后续可用 IC 加权 / ICIR 加权优化（本项目留了扩展点）
    """
    weights = weights or {"momentum": 0.4, "lowvol": 0.3, "reversal": 0.3}
    assert abs(sum(weights.values()) - 1.0) < 1e-9, "权重之和必须为1"

    factors = {
        "momentum": momentum_factor(close, 20),
        "lowvol": volatility_factor(close, 20),
        "reversal": reversal_factor(close, 5),
    }

    score = sum(weights[name] * f for name, f in factors.items())
    return score


def ic_series(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """计算逐日 IC（Information Coefficient）。

    IC = 当日因子值 与 次期收益率 的截面 Spearman 秩相关系数。

    解读：
    - IC > 0 且稳定：因子有预测力
    - |IC| < 0.02：基本没用
    - IC 均值 / IC 标准差 = ICIR，> 0.3 算不错的因子
    """
    aligned_f = factors.shift(1)  # 用 t-1 日的因子预测 t 日收益，避免前视偏差
    ics = []
    for date in forward_returns.index:
        f = aligned_f.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 5:
            continue
        ics.append(f[common].corr(r[common], method="spearman"))
    return pd.Series(ics, name="ic")


def factor_report(
    factor_values: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 5,
) -> dict:
    """生成单因子评价报告：IC 均值、ICIR、多空分组收益。"""
    fwd = close.pct_change(horizon).shift(-horizon)
    ic = ic_series(factor_values, fwd)
    ic = ic.dropna()

    if ic.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "ic_positive_rate": np.nan}

    ic_mean = float(ic.mean())
    ic_std = float(ic.std())
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": ic_mean / ic_std if ic_std > 0 else np.nan,
        "ic_positive_rate": float((ic > 0).mean()),
        "n_periods": int(len(ic)),
    }
