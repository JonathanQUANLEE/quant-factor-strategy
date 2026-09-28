"""数据模块：优先拉取真实行情，网络不可用时降级为合成数据。

设计考虑（面试可讲）：
1. 真实数据源（yfinance）在弱网环境下会失败，必须有 fallback，
   否则整个回测 pipeline 无法运行。
2. 合成数据不是随机噪声——我们注入了动量/低波动的收益结构，
   这样因子回测的结果方向是可解释的，方便单元测试验证逻辑正确性。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 沪深常见蓝筹 + 成长股代码（yfinance 格式）
DEFAULT_TICKERS = [
    "600519.SS",  # 贵州茅台
    "601318.SS",  # 中国平安
    "000858.SZ",  # 五粮液
    "300750.SZ",  # 宁德时代
    "600036.SS",  # 招商银行
    "000333.SZ",  # 美的集团
    "601899.SS",  # 紫金矿业
    "600276.SS",  # 恒瑞医药
    "002594.SZ",  # 比亚迪
    "601012.SS",  # 隆基绿能
    "600030.SS",  # 中信证券
    "000001.SZ",  # 平安银行
    "601166.SS",  # 兴业银行
    "600887.SS",  # 伊利股份
    "000651.SZ",  # 格力电器
    "601888.SS",  # 中国中免
    "600809.SS",  # 山西汾酒
    "002415.SZ",  # 海康威视
    "600585.SS",  # 海螺水泥
    "601398.SS",  # 工商银行
]


def fetch_real_data(
    tickers: list[str] | None = None,
    start: str = "2022-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame | None:
    """尝试通过 yfinance 拉取真实日线数据。

    返回 MultiIndex 列的 DataFrame（level0=字段, level1=代码），
    失败时返回 None（由调用方降级）。
    """
    tickers = tickers or DEFAULT_TICKERS
    try:
        import yfinance as yf

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw is None or raw.empty:
            return None

        close = raw["Close"].dropna(how="all")
        volume = raw["Volume"].dropna(how="all")
        high = raw["High"].dropna(how="all")
        low = raw["Low"].dropna(how="all")

        # 至少要有 15 只股票、250 个交易日才有回测意义
        if close.shape[1] < 10 or close.shape[0] < 250:
            return None

        return {"close": close, "volume": volume, "high": high, "low": low}
    except Exception:
        return None


def generate_synthetic_data(
    tickers: list[str] | None = None,
    n_days: int = 750,
    n_stocks: int = 40,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """生成带因子结构的合成日线数据。

    注入三种可被因子捕获的结构：
    - 动量：多数股票存在趋势延续（截面动量），persistence > 0
    - 低波动：高波动股票长期收益略差（低波动异象）
    - 反转：少数股票短期超涨后回调（persistence < 0）

    这样回测结果有可解释的方向，便于验证因子逻辑是否正确。
    """
    rng = np.random.default_rng(seed)
    tickers = tickers or DEFAULT_TICKERS
    # 扩展到 n_stocks 只（代码用序号命名）
    if len(tickers) < n_stocks:
        tickers = tickers + [f"STOCK{i:02d}.SS" for i in range(n_stocks - len(tickers))]
    tickers = tickers[:n_stocks]
    n_stocks = len(tickers)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    # 每只股票的特性：beta、特质波动、趋势持续性、动量强度
    market_ret = rng.normal(0.0003, 0.010, n_days)  # 日度市场收益（非累计值）
    beta = rng.uniform(0.6, 1.5, n_stocks)
    idio_vol = rng.uniform(0.006, 0.022, n_stocks)  # 特质波动（0.6%-2.2%/日）
    # 趋势持续性：多数 >0（动量延续），少数 <0（反转）—— 让动量因子有截面区分度
    persistence = rng.uniform(-0.10, 0.50, n_stocks)
    # 动量强度 + 低波动异象（高波动股票长期漂移明显更差）
    momentum_edge = (
        rng.uniform(-0.0008, 0.0012, n_stocks)
        - (idio_vol - idio_vol.mean()) * 0.06
    )

    log_returns = np.zeros((n_days, n_stocks))
    for t in range(n_days):
        # 趋势/反转持续性：昨日收益对今日的影响（按股票差异化）
        if t > 0:
            streak = log_returns[t - 1] * persistence
        else:
            streak = np.zeros(n_stocks)
        base = (
            beta * market_ret[t]
            + momentum_edge
            + rng.normal(0, 1, n_stocks) * idio_vol
            + streak
        )
        log_returns[t] = base

    # 从价格路径反推 OHLCV
    close = pd.DataFrame(
        np.exp(np.cumsum(log_returns, axis=0)) * 100.0,
        index=dates,
        columns=tickers,
    )
    daily_range = np.abs(rng.normal(0, 1, (n_days, n_stocks))) * 0.01 + 0.005
    high = close * (1 + daily_range / 2)
    low = close * (1 - daily_range / 2)
    volume = pd.DataFrame(
        rng.lognormal(12, 0.6, (n_days, n_stocks)),
        index=dates,
        columns=tickers,
    )

    # 模拟停牌：随机抹掉少量数据，测试数据清洗逻辑
    # 注意 pandas 3.0 Copy-on-Write 下 .values 为只读，需重建 DataFrame
    mask = rng.random((n_days, n_stocks)) < 0.01

    def apply_mask(df: pd.DataFrame) -> pd.DataFrame:
        arr = df.to_numpy(dtype=float, copy=True)
        arr[mask] = np.nan
        return pd.DataFrame(arr, index=df.index, columns=df.columns)

    close = apply_mask(close)
    high = apply_mask(high)
    low = apply_mask(low)
    volume = apply_mask(volume)

    return {
        "close": close,
        "volume": volume,
        "high": high,
        "low": low,
    }


def load_data(
    tickers: list[str] | None = None,
    start: str = "2022-01-01",
    end: str = "2025-12-31",
    allow_synthetic: bool = True,
) -> tuple[dict[str, pd.DataFrame], str]:
    """统一数据入口：真实数据优先，失败降级合成数据。

    Returns:
        (data_dict, source_tag)  source_tag ∈ {"real", "synthetic"}
    """
    real = fetch_real_data(tickers, start, end)
    if real is not None:
        return real, "real"

    if not allow_synthetic:
        raise RuntimeError("真实数据获取失败，且 allow_synthetic=False")

    print("[data] 真实行情获取失败（网络限流/不可达），降级为合成数据")
    return generate_synthetic_data(tickers), "synthetic"


def clean_data(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """数据清洗（面试高频考点）：
    1. 剔除缺失率 > 30% 的股票（疑似长期停牌/退市）
    2. 价格前向填充——绝不能 fillna(0)，价格为 0 会导致
       pct_change 产出 -100%/inf，污染整个回测
    3. 填充后仍缺数据的首段交易日直接剔除（不用 bfill，
       避免用未来数据填过去，造成前视偏差）
    4. 成交量缺失置 0（停牌日本来就无成交）
    """
    close = data["close"]

    # 1. 剔除缺失过多的股票
    valid_cols = close.columns[close.isna().mean() < 0.30]
    dropped = set(close.columns) - set(valid_cols)
    if dropped:
        print(f"[clean] 剔除缺失率>30%的股票: {sorted(dropped)}")

    cleaned = {k: v[valid_cols] for k, v in data.items()}

    # 2. 价格过滤：剔除非正价格，前向填充
    close = cleaned["close"].where(cleaned["close"] > 0)
    close = close.ffill()

    # 3. 填充后开头仍缺数据的行剔除（这些行无法计算收益）
    first_valid = close.apply(lambda s: s.first_valid_index())
    if first_valid.notna().any():
        # 任一股票仍缺数据的行 -> 剔除
        close = close.dropna(how="any")

    if close.empty:
        raise ValueError("数据清洗后为空，请检查输入数据质量")

    cleaned["close"] = close

    # 4. 其余字段对齐到 close 的 index/columns
    for key in ("high", "low", "volume"):
        if key in cleaned:
            df = cleaned[key].reindex(close.index, columns=close.columns)
            if key == "volume":
                df = df.fillna(0)
            else:
                df = df.ffill()
            cleaned[key] = df

    return cleaned
