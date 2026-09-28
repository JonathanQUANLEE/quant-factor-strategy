"""多因子选股量化策略——完整回测流水线。

运行：python main.py

流程：
1. 加载行情数据（真实优先 / 合成降级）
2. 数据清洗
3. 计算三个因子 + IC 分析
4. 合成综合打分
5. 向量化回测（含交易成本）
6. 计算绩效指标（夏普/回撤/卡玛）
7. 输出报告 + 保存图表
"""

from __future__ import annotations

import io
import os
import sys
import warnings

# Windows 控制台默认 GBK，强制 UTF-8 避免中文乱码
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")  # 无显示器环境也能出图
import matplotlib.pyplot as plt
import pandas as pd

from backtest import BacktestConfig, buy_and_hold, run_backtest
from data import clean_data, load_data
from factors import composite_score, factor_report, momentum_factor, reversal_factor, volatility_factor
from metrics import performance_summary

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 1. 数据 ----------
    print_header("STEP 1  数据加载")
    raw, source = load_data(allow_synthetic=True)
    print(f"数据来源: {source}")
    print(f"股票数: {raw['close'].shape[1]}  交易日数: {raw['close'].shape[0]}")
    print(f"时间范围: {raw['close'].index[0].date()} ~ {raw['close'].index[-1].date()}")

    # ---------- 2. 清洗 ----------
    print_header("STEP 2  数据清洗")
    data = clean_data(raw)
    close = data["close"]
    print(f"清洗后股票数: {close.shape[1]}")

    # ---------- 3. 因子计算 + IC ----------
    print_header("STEP 3  因子计算与 IC 分析")
    factor_values = {
        "动量 (20日)": momentum_factor(close, 20),
        "低波动 (20日)": volatility_factor(close, 20),
        "短期反转 (5日)": reversal_factor(close, 5),
    }

    print(f"\n{'因子':<16} {'IC均值':>10} {'ICIR':>10} {'IC>0占比':>10} {'样本数':>8}")
    print("-" * 60)
    for name, fval in factor_values.items():
        rep = factor_report(fval, close, horizon=5)
        print(
            f"{name:<16} {rep['ic_mean']:>10.4f} {rep['icir']:>10.3f} "
            f"{rep['ic_positive_rate']:>9.1%} {rep['n_periods']:>8d}"
        )

    # ---------- 4. 综合打分 ----------
    print_header("STEP 4  多因子合成")
    # 权重依据 STEP 3 的 IC 分析调整：
    # 动量/低波动 IC 为正 -> 给主要权重；反转 IC 为负 -> 降权观察
    weights = {"momentum": 0.45, "lowvol": 0.40, "reversal": 0.15}
    score = composite_score(close, weights)
    print(f"权重: {weights}")
    print(f"打分矩阵: {score.shape[0]} 日 × {score.shape[1]} 股票")

    # ---------- 5. 回测 ----------
    print_header("STEP 5  向量化回测")
    cfg = BacktestConfig(top_n=5, rebalance_days=20, cost_per_side=0.001)
    print(f"持仓数 top_n = {cfg.top_n}  调仓周期 = {cfg.rebalance_days} 日  单边成本 = {cfg.cost_per_side:.2%}")

    result = run_backtest(score, close, cfg)
    benchmark = buy_and_hold(close)  # 返回净值曲线
    # performance_summary 需要日收益序列，净值要先转回日收益
    benchmark_daily = benchmark.pct_change().fillna(0.0)

    # ---------- 6. 绩效 ----------
    print_header("STEP 6  绩效指标")
    summary = performance_summary(result["daily_returns"], benchmark_daily)
    print(summary.to_string(index=False))

    avg_turnover = result["turnover"].mean()
    print(f"\n平均单次换手率: {avg_turnover:.2%}")

    # ---------- 7. 图表 ----------
    print_header("STEP 7  生成报告图表")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (1) 净值曲线
    ax = axes[0, 0]
    ax.plot(result["nav"].index, result["nav"].values, label="多因子策略", linewidth=1.5)
    ax.plot(benchmark.index, benchmark.values, label="等权基准", linewidth=1.0, alpha=0.7)
    ax.set_title("累计净值对比")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) 回撤曲线
    ax = axes[0, 1]
    nav = result["nav"]
    dd = nav / nav.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.35)
    ax.set_title("策略回撤")
    ax.grid(alpha=0.3)

    # (3) 月度收益热力图
    ax = axes[1, 0]
    monthly = (1 + result["daily_returns"]).resample("ME").prod() - 1
    try:
        pivot = pd.DataFrame(
            {
                "year": monthly.index.year,
                "month": monthly.index.month,
                "ret": monthly.values,
            }
        ).pivot_table(index="year", columns="month", values="ret")
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels([f"{m}月" for m in pivot.columns], fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title("月度收益热力图")
        from matplotlib.ticker import FuncFormatter

        fig.colorbar(im, ax=ax, format=FuncFormatter(lambda v, _: f"{v:.0%}"))
    except Exception as e:
        ax.set_title(f"月度收益（生成失败: {e}）")

    # (4) 换手率
    ax = axes[1, 1]
    ax.bar(
        range(len(result["turnover"])),
        result["turnover"].values,
        width=1.0,
        color="steelblue",
    )
    ax.set_title(f"每次调仓换手率（均值 {avg_turnover:.1%}）")
    ax.grid(alpha=0.3)

    fig.suptitle(f"多因子选股策略回测报告（数据源: {source}）", fontsize=14)
    fig.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "backtest_report.png")
    fig.savefig(chart_path, dpi=120)
    print(f"图表已保存: {chart_path}")

    # ---------- 8. 导出数据 ----------
    csv_path = os.path.join(OUTPUT_DIR, "performance_summary.csv")
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    nav_path = os.path.join(OUTPUT_DIR, "nav_curve.csv")
    pd.DataFrame(
        {"strategy": result["nav"], "benchmark": benchmark.reindex(result["nav"].index).ffill()}
    ).to_csv(nav_path, encoding="utf-8-sig")

    print(f"绩效表已保存: {csv_path}")
    print(f"净值曲线已保存: {nav_path}")

    print_header("完成")
    print("面试讲述要点见 README.md")


if __name__ == "__main__":
    main()
