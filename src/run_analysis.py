"""Run the full analysis: panel -> rule paths -> sensitivity tables -> charts -> results.md.

Usage:  python src/run_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import build_panel
from taylor import INERTIAL, TAYLOR_1993, TAYLOR_1999, taylor_rate

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "results", ROOT / "figures"

# ---- assumptions --------------------------------------------------------------------------
BASE_R_STAR = 2.0      # Taylor's (1993) original real neutral rate
R_STARS = [2.0, 1.5, 1.0, 0.5, 0.0, -0.5]
GAPS = {"gap_obr": "OBR (Oct 2026 EFO)", "gap_hp": "HP filter (two-sided)",
        "gap_hp_realtime": "HP filter (real-time)", "gap_hp_excovid": "HP filter (ex-Covid trend)",
        "gap_zero": "Zero gap"}
INFLATION = {"cpi": "Headline CPI", "core_cpi": "Core CPI"}
RULES = [TAYLOR_1993, TAYLOR_1999]

# ---- chart style (validated default palette, categorical slots in fixed order) -------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
})


def ts(idx: pd.PeriodIndex):
    return idx.to_timestamp()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    p = build_panel()

    # --- 1. baseline rule and headline gap in 2022 ---------------------------------------
    base = taylor_rate(p["cpi"], p["gap_obr"], BASE_R_STAR, TAYLOR_1993)
    gap = (base - p["bank_rate"]).rename("rule_minus_actual")
    y22 = gap.loc["2022-01":"2022-12"]
    hm = y22.idxmax()  # month in 2022 where the shortfall peaked
    X, Y, Z = base[hm], p.loc[hm, "bank_rate"], gap[hm]

    paths = pd.DataFrame({"bank_rate": p["bank_rate"], "taylor_1993_baseline": base,
                          "shortfall_pp": gap})
    for r in R_STARS:
        paths[f"taylor_1993_rstar_{r:+.1f}"] = taylor_rate(p["cpi"], p["gap_obr"], r)
    paths["taylor_1993_core_rstar_0.5"] = taylor_rate(p["core_cpi"], p["gap_obr"], 0.5)
    paths["inertial_rstar_0.5"] = taylor_rate(p["cpi"], p["gap_obr"], 0.5, INERTIAL, actual=p["bank_rate"])
    paths.round(3).to_csv(OUT / "rule_paths.csv")

    # --- 2. sensitivity grid at the headline month ----------------------------------------
    rows = []
    for rule in RULES:
        for inf, inf_lab in INFLATION.items():
            for g, g_lab in GAPS.items():
                for r in R_STARS:
                    i = taylor_rate(p[inf], p[g], r, rule)[hm]
                    rows.append(dict(rule=rule.name, inflation=inf_lab, output_gap=g_lab, r_star=r,
                                     implied=i, actual=Y, shortfall=i - Y))
    grid = pd.DataFrame(rows)
    grid.round(2).to_csv(OUT / "sensitivity_grid.csv", index=False)

    # --- 3. decompose the baseline shortfall -------------------------------------------
    pi, og = p.loc[hm, "cpi"], p.loc[hm, "gap_obr"]
    decomp = pd.Series({
        "Neutral real rate r* (2%)": BASE_R_STAR,
        "Inflation (pass-through, 1×π)": pi,
        "Inflation deviation (0.5×(π−2))": 0.5 * (pi - 2),
        "Output gap (0.5×gap)": 0.5 * og,
        "Actual Bank Rate": -Y,
    })
    # how much of the shortfall does the r* assumption explain?
    r_share = {r: (BASE_R_STAR - r) / Z for r in R_STARS if r < BASE_R_STAR}
    core_alt = taylor_rate(p["core_cpi"], p["gap_obr"], 0.5)[hm] - Y
    t93 = grid[grid.rule == TAYLOR_1993.name]
    span = t93[t93.inflation == "Headline CPI"]["shortfall"]
    min_any = grid["shortfall"].min()

    # 'behind the curve' over the whole sample: months where actual < every r*-scenario
    low_rule = taylor_rate(p["cpi"], p["gap_obr"], min(R_STARS))
    behind = (p["bank_rate"] < low_rule)
    behind_core = p["bank_rate"] < taylor_rate(p["core_cpi"], p["gap_obr"], min(R_STARS))
    first_gt5 = gap[gap > 5].index.min()

    # --- 4. charts -----------------------------------------------------------------------
    t = ts(p.index)
    # (a) full history
    fig, ax = plt.subplots(figsize=(10, 5.2))
    lo = taylor_rate(p["cpi"], p["gap_obr"], 0.0)
    ax.fill_between(t, lo, base, color=S[0], alpha=0.18, lw=0, label="Taylor (1993), r* from 0% to 2%")
    ax.plot(t, base, color=S[0], label="Taylor (1993), r* = 2% (baseline)")
    ax.plot(t, paths["taylor_1993_core_rstar_0.5"], color=S[1], lw=1.6, label="Taylor (1993), core CPI, r* = 0.5%")
    ax.plot(t, p["bank_rate"], color=INK, lw=2.2, label="Actual Bank Rate")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.annotate(f"{hm.strftime('%b %Y')}: rule {X:.1f}% vs actual {Y:.2f}%",
                xy=(hm.to_timestamp(), X), xytext=(pd.Timestamp("2012-06-01"), X - 1.5),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), color=INK, fontsize=9)
    ax.set_title("UK Bank Rate vs Taylor-rule prescriptions, 1997–2026")
    ax.set_ylabel("Per cent")
    ax.legend(loc="upper left", fontsize=9)
    fig.text(0.01, 0.01, "Sources: Bank of England (IUDBEDR), ONS (D7G7, DKO8), OBR public finances databank. "
             "Output gap: OBR. Rule: i = r* + π + 0.5(π − 2) + 0.5·gap.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "01_rule_vs_bank_rate_1997_2026.png", dpi=200)
    plt.close(fig)

    # (b) zoom 2020-2026 with variants
    z = slice("2020-01", None)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    pz = p.loc[z]
    tz = ts(pz.index)
    ax.plot(tz, base.loc[z], color=S[0], label="Taylor (1993), headline CPI, r* = 2%")
    ax.plot(tz, taylor_rate(pz["cpi"], pz["gap_obr"], 0.0), color=S[0], lw=1.4, ls="--",
            label="Taylor (1993), headline CPI, r* = 0%")
    ax.plot(tz, paths.loc[z, "taylor_1993_core_rstar_0.5"], color=S[1], label="Taylor (1993), core CPI, r* = 0.5%")
    ax.plot(tz, paths.loc[z, "inertial_rstar_0.5"], color=S[2], label="Inertial rule (ρ = 0.85/qtr), r* = 0.5%")
    ax.plot(tz, pz["bank_rate"], color=INK, lw=2.2, label="Actual Bank Rate")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("The 2021–23 inflation shock: how far behind was Bank Rate?")
    ax.set_ylabel("Per cent")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "02_zoom_2020_2026.png", dpi=200)
    plt.close(fig)

    # (c) heatmap: shortfall at headline month, r* x output-gap measure (Taylor 1993, headline CPI)
    hmap = t93[t93.inflation == "Headline CPI"].pivot(index="output_gap", columns="r_star", values="shortfall")
    hmap = hmap.loc[list(GAPS.values()), sorted(R_STARS, reverse=True)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(hmap.values, cmap="Blues", aspect="auto", vmin=0)
    ax.grid(False)
    ax.set_xticks(range(hmap.shape[1]), [f"{c:+.1f}%" for c in hmap.columns])
    ax.set_yticks(range(hmap.shape[0]), hmap.index)
    ax.set_xlabel("Neutral real rate r*")
    for (i, j), v in np.ndenumerate(hmap.values):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                color="white" if v > hmap.values.max() * 0.6 else INK)
    ax.set_title(f"Rule minus actual Bank Rate in {hm.strftime('%b %Y')} (pp), Taylor (1993), headline CPI")
    fig.colorbar(im, ax=ax, shrink=0.8, label="pp")
    fig.tight_layout()
    fig.savefig(FIG / "03_sensitivity_heatmap.png", dpi=200)
    plt.close(fig)

    # (d) decomposition
    fig, ax = plt.subplots(figsize=(9, 4))
    vals = decomp.values
    cum = np.concatenate([[0], np.cumsum(vals)[:-1]])
    colors = [S[0], S[0], S[0], S[0], INK2]
    ax.bar(range(len(vals)), vals, bottom=cum, color=colors, width=0.6, edgecolor=SURF, linewidth=2)
    ax.bar(len(vals), Z, color=S[1], width=0.6)
    for k, (b, v) in enumerate(zip(cum, vals)):
        ax.text(k, max(b, b + v) + 0.3, f"{v:+.2f}", ha="center", fontsize=9)
    ax.text(len(vals), Z + 0.3, f"{Z:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(vals) + 1), list(decomp.index) + ["Shortfall"], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Percentage points")
    ax.set_ylim(0, max(np.cumsum(vals)) * 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"What drives the {hm.strftime('%b %Y')} shortfall? (baseline rule)")
    fig.tight_layout()
    fig.savefig(FIG / "04_shortfall_decomposition.png", dpi=200)
    plt.close(fig)

    # (e) output-gap measures
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, g in enumerate(["gap_obr", "gap_hp", "gap_hp_realtime", "gap_hp_excovid"]):
        ax.plot(t, p[g], color=S[k], lw=1.6, label=GAPS[g])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(-8, 6)
    ax.set_title("Output-gap estimates (% of potential GDP); 2020 Covid trough clipped")
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "05_output_gap_measures.png", dpi=200)
    plt.close(fig)

    # --- 5. results.md --------------------------------------------------------------------
    by_gap = t93[(t93.inflation == "Headline CPI") & (t93.r_star == BASE_R_STAR)].set_index("output_gap")["shortfall"]
    lines = [
        "# Results (generated by src/run_analysis.py)", "",
        f"Sample: {p.index[0]} to {p.index[-1]} (monthly; last month with both CPI and Bank Rate).", "",
        "## Headline (baseline: Taylor 1993, headline CPI, r* = 2%, OBR output gap)", "",
        f"* Peak 2022 shortfall month: **{hm.strftime('%B %Y')}**",
        f"* Rule-implied Bank Rate: **{X:.1f}%** (CPI {pi:.1f}%, OBR gap {og:+.2f}%)",
        f"* Actual Bank Rate (month-end): **{Y:.2f}%**",
        f"* Gap: **{Z:.1f}pp**",
        f"* First month the baseline shortfall exceeded 5pp: {first_gt5.strftime('%B %Y')}", "",
        "## How much does the neutral-rate assumption explain?", "",
        *[f"* Cutting r* from 2% to {r:+.1f}% closes {BASE_R_STAR - r:.1f}pp = {s:.0%} of the {Z:.1f}pp gap"
          for r, s in r_share.items()],
        f"* Across all r* (−0.5% to 2%) and all five gap measures (Taylor 1993, headline CPI) the shortfall ranges "
        f"{span.min():.1f}–{span.max():.1f}pp.",
        f"* Using core CPI with r* = 0.5% still leaves a {core_alt:.1f}pp shortfall.",
        f"* Smallest shortfall in the whole grid (any rule, inflation measure, gap, r*): {min_any:.1f}pp.", "",
        "## Shortfall by output-gap measure (r* = 2%)", "",
        *[f"* {k}: {v:.1f}pp" for k, v in by_gap.items()], "",
        "## Behind the curve over the whole sample", "",
        f"* Months with Bank Rate below the Taylor (1993) rule even at r* = {min(R_STARS)}% (headline CPI): "
        f"{int(behind.sum())} of {len(p)}; using core CPI: {int(behind_core.sum())}.",
        f"* Mean shortfall (baseline) 1997-05 to 2007-12: {gap.loc[:'2007-12'].mean():.2f}pp; "
        f"2009-03 to 2019-12: {gap.loc['2009-03':'2019-12'].mean():.2f}pp; "
        f"2021-01 to 2023-12: {gap.loc['2021-01':'2023-12'].mean():.2f}pp; "
        f"2024-01 to end: {gap.loc['2024-01':].mean():.2f}pp.",
    ]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    decomp.round(3).to_csv(OUT / "shortfall_decomposition.csv", header=["pp"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
