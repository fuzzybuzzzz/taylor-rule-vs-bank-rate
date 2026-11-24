"""Clean and align the raw series into one monthly panel (data/processed/panel.csv)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"


def hp_filter(y: np.ndarray, lam: float = 1600.0) -> np.ndarray:
    """Hodrick-Prescott trend (two-sided). Solves (I + lam*D'D) tau = y."""
    n = len(y)
    d = np.zeros((n - 2, n))
    for i in range(n - 2):
        d[i, i:i + 3] = [1.0, -2.0, 1.0]
    return np.linalg.solve(np.eye(n) + lam * d.T @ d, y)


def hp_one_sided(y: np.ndarray, lam: float = 1600.0, min_obs: int = 40) -> np.ndarray:
    """'Real-time' HP trend: the trend at t uses only data up to t (re-estimated each quarter)."""
    out = np.full(len(y), np.nan)
    for t in range(min_obs - 1, len(y)):
        out[t] = hp_filter(y[: t + 1], lam)[-1]
    return out


def load_gdp_gaps() -> pd.DataFrame:
    g = pd.read_csv(RAW / "ons_gdp_abmi.csv")
    g.index = pd.PeriodIndex(g["quarter"], freq="Q")
    y = np.log(g["real_gdp_ABMI_gbp_m"].to_numpy()) * 100
    out = pd.DataFrame(index=g.index)
    out["gap_hp"] = y - hp_filter(y)
    out["gap_hp_realtime"] = y - hp_one_sided(y)
    # Covid (2020Q1-2021Q3) drags any statistical trend down and inflates the 2021-22 gap.
    # Variant: estimate the trend on a series where those quarters are linearly interpolated,
    # then measure actual GDP against that trend.
    yc = pd.Series(y, index=g.index)
    covid = (yc.index >= pd.Period("2020Q1", "Q")) & (yc.index <= pd.Period("2021Q3", "Q"))
    yc[covid] = np.nan
    yc = yc.interpolate()
    out["gap_hp_excovid"] = y - hp_filter(yc.to_numpy())
    return out


def quarterly_to_monthly(q: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    m = q.copy()
    m.index = m.index.asfreq("M", how="end") - 1  # place each quarter at its middle month
    m = m.reindex(pd.period_range(m.index.min(), max(months.max(), m.index.max()), freq="M"))
    m = m.interpolate(limit_area="inside").ffill()  # carry the last estimate forward
    return m.reindex(months)


def load_obr_gap(months: pd.PeriodIndex) -> pd.Series:
    """OBR fiscal-year output gap -> monthly by placing each April-March year at its midpoint
    (October) and interpolating linearly."""
    o = pd.read_csv(RAW / "obr_output_gap_fy.csv")
    start_year = o["fiscal_year"].str[:4].astype(int)
    s = pd.Series(o["output_gap_pct"].to_numpy(), index=pd.PeriodIndex([f"{y}-10" for y in start_year], freq="M"))
    full = pd.period_range(s.index.min(), s.index.max(), freq="M")
    return s.reindex(full).interpolate().reindex(months).rename("gap_obr")


def build_panel(start: str = "1997-05", save: bool = True) -> pd.DataFrame:
    br = pd.read_csv(RAW / "bank_rate_monthend.csv")
    br.index = pd.PeriodIndex(br["month"], freq="M")
    cpi = pd.read_csv(RAW / "ons_cpi.csv")
    cpi.index = pd.PeriodIndex(cpi["month"], freq="M")

    end = min(br.index.max(), cpi.index.max())  # last month with both Bank Rate and CPI
    months = pd.period_range(start, end, freq="M")
    p = pd.DataFrame(index=months)
    p["bank_rate"] = br["bank_rate"].reindex(months)
    p["cpi"] = cpi["cpi_yoy_D7G7"].reindex(months)
    p["core_cpi"] = cpi["core_cpi_yoy_DKO8"].reindex(months)
    p["gap_obr"] = load_obr_gap(months)
    p = p.join(quarterly_to_monthly(load_gdp_gaps(), months))
    p["gap_zero"] = 0.0
    p.index.name = "month"
    assert p[["bank_rate", "cpi", "core_cpi", "gap_obr"]].notna().all().all(), "missing data in panel"
    if save:
        PROC.mkdir(parents=True, exist_ok=True)
        p.round(4).to_csv(PROC / "panel.csv")
    return p


if __name__ == "__main__":
    print(build_panel().tail())
