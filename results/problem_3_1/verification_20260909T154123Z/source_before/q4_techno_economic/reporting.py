"""Machine-readable comparisons and compact plots; no unsupported investment claims."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .finance import pareto_flags


def build_comparisons(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.copy()
    summary["pareto_within_technology"] = False
    rows = []
    for tech, group in summary.groupby("technology"):
        vals = group["lifetime_net_renewable_gain_mwh"].to_numpy()
        returns = group["npv_eur"].to_numpy()
        # Include no-build when assessing dominance and the commercial decision.
        keep = pareto_flags(np.r_[vals, 0.0], np.r_[returns, 0.0])[:-1]
        summary.loc[group.index, "pareto_within_technology"] = keep
        technical = group.loc[group.lifetime_net_renewable_gain_mwh.idxmax()]
        best = group.loc[group.npv_eur.idxmax()]
        for label, selected in (("technical_max_net_RE_gain", technical), ("economic_max_conditional_NPV", best)):
            no_build = (label.startswith("economic") and selected.npv_eur <= 0) or (
                label.startswith("technical") and selected.lifetime_net_renewable_gain_mwh <= 0)
            rows.append({"technology": tech, "selection": label,
                "case_id": "NO_BUILD" if no_build else selected.case_id,
                "bus": "-" if no_build else selected.bus,
                "wind_mw": 0 if no_build else selected.wind_mw,
                "battery_mw": 0 if no_build else selected.battery_mw,
                "battery_mwh": 0 if no_build else selected.battery_mwh,
                "npv_eur": 0 if no_build else selected.npv_eur,
                "lifetime_net_renewable_gain_mwh": 0 if no_build else selected.lifetime_net_renewable_gain_mwh,
                "interpretation": "best among evaluated designs under stated operator dispatch and assumptions"})
    return summary, pd.DataFrame(rows)


def write_report(out: Path, summary: pd.DataFrame, config: dict, metadata: dict,
                 failures: list[dict], planned: int, smoke: bool) -> None:
    if summary.empty:
        (out / "SUMMARY.md").write_text("# Q4\n\nNo successful cases. Inspect failures.json and the terminal output.\n", encoding="utf-8")
        return
    summary, comparisons = build_comparisons(summary)
    summary.sort_values(["technology", "npv_eur"], ascending=[True, False]).to_csv(out / "01_project_ranking.csv", index=False)
    comparisons.to_csv(out / "02_technical_vs_economic.csv", index=False)
    summary.loc[summary.pareto_within_technology].to_csv(out / "03_pareto_frontier.csv", index=False)
    for tech in summary.technology.unique():
        frame = summary.loc[summary.technology.eq(tech)]
        frame.to_csv(out / f"ranking_{tech}.csv", index=False)
    lines = ["# Q4 技术—经济选址分析", "",
        "**本文件是情景模拟，不是实际投资建议或并网承诺。**", "",
        f"- Backend: `{metadata['backend']}`",
        f"- Cases completed: {len(summary)} / {planned}; failed: {len(failures)}",
        f"- Financial horizon: {config['years']} years; real discount rate: {config['real_discount_rate']:.1%}",
        f"- Currency: {config['currency_basis']}",
        f"- Assumptions status: {config['assumptions_status']}",
        f"- Annualisation: {config['annualisation']}", "",
        "SMOKE TEST ONLY — 缩短的时间序列和寿命不能用于投资结论。" if smoke else "",
        "## 比较口径", "",
        "技术最优：已测试候选中，寿命期内新增风光利用量减去项目电池损耗的代理指标最大。",
        "它不是逐 MWh 来源追踪，也不是包含已有储能/可控联络线损耗的系统总能量认证。",
        "经济最优：在相同电网调度规则下，候选生命周期 NPV 最大；所有 NPV≤0 时选择不投资。",
        "NPV 不是开发商自主竞价/套利调度的全局最优。两种最优都仅限已测试离散候选。", "",
        "## 结果", ""]
    for row in comparisons.to_dict("records"):
        lines.append(f"**{row['technology']} / {row['selection']}**: `{row['case_id']}` @ `{row['bus']}`, NPV €{row['npv_eur']:,.0f}.")
        lines.append("")
    lines += ["## 必须保留的限制", "",
        "1. 官方网络的 DC 潮流、边界和并行回路原样保留；Q4 不会替代 15/16-node/all-island 验证。",
        "2. 合成代表周重复至一年只是情景外推。更换成真实电价不等于合成风电和负荷变成真实联合时间序列。",
        "3. 电池每年重跑容量受限的调度；年内容量按年初值保持，退化和维护发生在年末。",
        "4. 无税务/债务/补贴/真实系统服务竞标/电价内生反馈/AC电压/N−1验证；寿命内网络拓扑与背景机组不变。",
        "5. 新增风电默认使用共同风况以隔离电网位置效应；不是各站真实风资源评估。共站风场按可用功率同比分配，未模拟优先权。",
        "6. LCOS 仅对独立电池；混合项目采用共同计量点净结算，不把内部充电当成一次销售。",
        "7. 运维不含电芯扩容/更换，相关支出另列；调度 wear penalty 不再次作为现金成本扣除。",
        "8. 敏感性表是固定物理运行路径重估；改变退化/效率/寿命须修改配置重新运行。",
        "9. first-year constraint split 是放宽 Line/Transformer 热限的反事实；不等于 SNSP curtailment。", "",
        "各方案逐年现金流、第一年逐小时功率/SoC、全网络分支指标见 `cases/<case_id>/`。",
        "输入、版本、数据指纹见 `manifest.json`；失败方案不进入排名。"]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    plot_outputs(out, summary, is_demo="demo" in metadata["backend"])


def plot_outputs(out: Path, summary: pd.DataFrame, is_demo: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title_prefix = "SYNTHETIC DEMO - NOT IRELAND\n" if is_demo else ""
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    for tech, group in summary.groupby("technology"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(group.lifetime_net_renewable_gain_mwh / 1e3, group.npv_eur / 1e6)
        ax.scatter([0], [0], marker="x", label="No build")
        frontier = group.loc[group.pareto_within_technology].sort_values("lifetime_net_renewable_gain_mwh")
        if len(frontier) > 1:
            ax.plot(frontier.lifetime_net_renewable_gain_mwh / 1e3, frontier.npv_eur / 1e6, linestyle="--", label="Evaluated Pareto frontier")
        for _, row in group.nlargest(3, "npv_eur").iterrows():
            label = f"{row.bus} | W{row.wind_mw:g} B{row.battery_mw:g}/{row.battery_mwh:g}"
            ax.annotate(label, (row.lifetime_net_renewable_gain_mwh / 1e3, row.npv_eur / 1e6),
                        xytext=(-4, 5) if row.lifetime_net_renewable_gain_mwh > group.lifetime_net_renewable_gain_mwh.median() else (4, 5),
                        ha="right" if row.lifetime_net_renewable_gain_mwh > group.lifetime_net_renewable_gain_mwh.median() else "left",
                        textcoords="offset points", fontsize=7)
        ax.axhline(0, linewidth=0.7)
        ax.set(xlabel="Lifetime net renewable gain proxy (GWh)", ylabel="Conditional project NPV (EUR million)",
               title=title_prefix + f"{tech}: technical benefit vs economic return - scenario model")
        ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(directory / f"pareto_{tech}.png", dpi=150); plt.close(fig)
        best = group.loc[group.npv_eur.idxmax()]
        path = out / "cases" / best.case_id / "annual_cashflows.csv"
        cf = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(cf.year, cf.cumulative_cashflow_eur / 1e6, label="Undiscounted")
        ax.plot(cf.year, cf.cumulative_discounted_cashflow_eur / 1e6, label="Discounted")
        ax.axhline(0, linewidth=0.7)
        ax.set(xlabel="Project year", ylabel="Cumulative cash flow (EUR million)",
               title=title_prefix + f"Best evaluated {tech} candidate - compare with no build")
        ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(directory / f"cashflow_{tech}.png", dpi=150); plt.close(fig)

        sensitivity_path = path.parent / "financial_sensitivity.csv"
        if sensitivity_path.exists():
            sens = pd.read_csv(sensitivity_path)
            ranges = sens.groupby("parameter").npv_eur.agg(["min", "max"])
            ranges = ranges.loc[(ranges["max"] - ranges["min"]).sort_values().index]
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.barh(ranges.index, (ranges["max"] - ranges["min"]) / 1e6,
                    left=ranges["min"] / 1e6)
            ax.axvline(best.npv_eur / 1e6, linestyle="--", label="Base NPV")
            ax.set(xlabel="Conditional NPV range (EUR million)",
                   title=title_prefix + f"{tech}: fixed-dispatch financial sensitivity")
            ax.legend(); ax.grid(axis="x", alpha=0.25); fig.tight_layout()
            fig.savefig(directory / f"sensitivity_{tech}.png", dpi=150); plt.close(fig)
