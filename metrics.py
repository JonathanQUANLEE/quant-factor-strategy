"""绩效指标模块——回测报告的核心。

面试必问指标及标准答案（见每个函数 docstring）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(nav: pd.Series) -> float:
    """年化收益率。

    公式：(末值/初值)^(252/n交易日) - 1
    面试追问：为什么不直接用 mean*252？
    答：那是算术近似，复利场景下几何年化更准确，长周期差异明显。
    """
    if len(nav) < 2 or nav.iloc[0] == 0:
        return np.nan
    total = nav.iloc[-1] / nav.iloc[0]
    n_years = len(nav) / TRADING_DAYS
    return total ** (1 / n_years) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    """年化波动率 = 日收益标准差 × sqrt(252)。"""
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> float:
    """夏普比率（面试第一高频指标）。

    公式：(年化收益 - 无风险利率) / 年化波动率

    含义：每承担 1 单位波动风险，获得多少超额收益。
    - < 0:  还不如买货币基金
    - 0-1:  一般
    - 1-2:  不错
    - > 2:  很强（要警惕过拟合）

    追问：为什么用波动率做分母，不用最大回撤？
    答：波动率是二阶矩、可微、统计性质稳定；回撤是路径依赖的极值统计量，
    样本量小、方差大，不适合作为优化目标。
    """
    ann_ret = annualized_return(nav_from_returns(daily_returns))
    ann_vol = annualized_volatility(daily_returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol


def nav_from_returns(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns.fillna(0)).cumprod()


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（面试第二高频指标）。

    公式：min(净值 / 历史最高净值) - 1

    含义：历史上任意时点买入，最惨会亏多少。
    - 10% 以内：风控良好
    - 20% 以内：可接受
    - > 30%：对权益策略来说偏高

    与夏普的互补性：夏普惩罚对称波动，回撤惩罚单边深坑。
    一个策略可能夏普不错但回撤大（慢涨急跌），必须同时看。
    """
    if nav.empty:
        return np.nan
    cum_max = nav.cummax()
    dd = nav / cum_max - 1
    return float(dd.min())


def calmar_ratio(daily_returns: pd.Series) -> float:
    """卡玛比率 = 年化收益 / |最大回撤|。

    含义：每承受 1% 的最大回撤，获得多少年化收益。
    比夏普更关注尾部风险，> 1 算健康。
    """
    nav = nav_from_returns(daily_returns)
    ann_ret = annualized_return(nav)
    mdd = max_drawdown(nav)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return ann_ret / abs(mdd)


def win_rate(daily_returns: pd.Series) -> float:
    """日胜率 = 盈利天数 / 有交易的天数。

    注意：胜率高不等于赚钱——盈亏比同样重要。
    一个策略可以 80% 胜率但亏钱（小赚大亏）。
    """
    r = daily_returns[daily_returns != 0]
    if r.empty:
        return np.nan
    return float((r > 0).mean())


def profit_factor(daily_returns: pd.Series) -> float:
    """盈亏比 = 总盈利 / 总亏损（绝对值）。

    > 1 才赚钱；与胜率相乘的期望值才是真正的 edge。
    """
    gains = daily_returns[daily_returns > 0].sum()
    losses = -daily_returns[daily_returns < 0].sum()
    if losses == 0:
        return np.inf
    return float(gains / losses)


def performance_summary(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """一键生成绩效汇总表。"""
    nav = nav_from_returns(daily_returns)
    rows = {
        "年化收益率": f"{annualized_return(nav):.2%}",
        "年化波动率": f"{annualized_volatility(daily_returns):.2%}",
        "夏普比率": f"{sharpe_ratio(daily_returns):.2f}",
        "最大回撤": f"{max_drawdown(nav):.2%}",
        "卡玛比率": f"{calmar_ratio(daily_returns):.2f}",
        "日胜率": f"{win_rate(daily_returns):.2%}",
        "盈亏比": f"{profit_factor(daily_returns):.2f}",
        "累计净值": f"{nav.iloc[-1]:.4f}",
    }

    if benchmark_returns is not None:
        bm_nav = nav_from_returns(benchmark_returns)
        rows["基准年化"] = f"{annualized_return(bm_nav):.2%}"
        rows["超额收益(年化)"] = (
            f"{annualized_return(nav) - annualized_return(bm_nav):.2%}"
        )
        # 信息比率：超额收益的稳定性
        active = daily_returns - benchmark_returns
        te = active.std() * np.sqrt(TRADING_DAYS)
        if te > 0:
            excess = annualized_return(nav) - annualized_return(bm_nav)
            rows["信息比率"] = f"{excess / te:.2f}"

    return pd.DataFrame(list(rows.items()), columns=["指标", "数值"])
