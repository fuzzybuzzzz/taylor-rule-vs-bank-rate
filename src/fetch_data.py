"""Download raw data from the Bank of England, ONS and OBR into data/raw/.

Sources
-------
* Bank of England IADB, series IUDBEDR (official Bank Rate, daily) -> month-end values
* ONS MM23 D7G7 (CPI annual rate) and DKO8 (CPI excl. energy, food, alcohol & tobacco annual rate)
* ONS QNA ABMI (real GDP, chained volume measure, quarterly) -> used for HP-filter output gaps
* OBR Public finances databank (latest EFO), "Output gap (per cent of GDP)", fiscal years

The snapshots shipped in data/raw/ were downloaded on 24 Nov 2026. Re-running this
script overwrites them with whatever the sources publish today.

Usage:  python src/fetch_data.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (taylor-rule research script)"}

BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    "?csv.x=yes&Datefrom=01/Jan/1997&Dateto=now&SeriesCodes=IUDBEDR"
    "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)
ONS_URL = "https://www.ons.gov.uk/{path}/timeseries/{cdid}/{dataset}/data"
OBR_DATABANK_URL = "https://obr.uk/download/public-finances-databank-october-2026-efo/"


def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def fetch_bank_rate() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(_get(BOE_URL).text))
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d %b %Y")
    s = df.set_index("DATE")["IUDBEDR"].dropna()
    m = s.groupby(s.index.to_period("M")).last()
    out = pd.DataFrame({"month": m.index.astype(str), "bank_rate": m.values})
    out.to_csv(RAW / "bank_rate_monthend.csv", index=False)
    return out


def _ons_months(path: str, cdid: str, dataset: str) -> pd.Series:
    js = _get(ONS_URL.format(path=path, cdid=cdid, dataset=dataset)).json()
    idx = pd.to_datetime([m["date"].title() for m in js["months"]], format="%Y %b").to_period("M")
    return pd.Series([float(m["value"]) for m in js["months"]], index=idx)


def fetch_cpi() -> pd.DataFrame:
    p = "economy/inflationandpriceindices"
    head = _ons_months(p, "d7g7", "mm23")
    core = _ons_months(p, "dko8", "mm23")
    df = pd.concat({"cpi_yoy_D7G7": head, "core_cpi_yoy_DKO8": core}, axis=1)
    df = df[df.index >= pd.Period("1996-01", "M")]
    df.index = df.index.astype(str)
    df.index.name = "month"
    df.to_csv(RAW / "ons_cpi.csv")
    return df


def fetch_gdp() -> pd.DataFrame:
    js = _get(ONS_URL.format(path="economy/grossdomesticproductgdp", cdid="abmi", dataset="qna")).json()
    rows = [(q["date"].replace(" ", ""), float(q["value"])) for q in js["quarters"] if int(q["year"]) >= 1985]
    df = pd.DataFrame(rows, columns=["quarter", "real_gdp_ABMI_gbp_m"])
    df.to_csv(RAW / "ons_gdp_abmi.csv", index=False)
    return df


def fetch_obr_output_gap(url: str = OBR_DATABANK_URL) -> pd.DataFrame:
    """Read 'Output gap (per cent of GDP)' from the OBR public finances databank."""
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(_get(url).content), data_only=True, read_only=True)
    ws = wb["Aggregates (per cent of GDP)"]
    rows = list(ws.iter_rows(values_only=True))
    hdr_r, col = next(
        (r, c) for r, row in enumerate(rows) for c, v in enumerate(row)
        if isinstance(v, str) and v.strip().startswith("Output gap")
    )
    out = []
    for row in rows[hdr_r + 1:]:
        fy = row[1] or row[0]
        val = row[col]
        if isinstance(fy, str) and "-" in fy and isinstance(val, (int, float)):
            out.append((fy.strip(), round(float(val), 2)))
    df = pd.DataFrame(out, columns=["fiscal_year", "output_gap_pct"])
    df.to_csv(RAW / "obr_output_gap_fy.csv", index=False)
    return df


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fn in [("Bank Rate", fetch_bank_rate), ("CPI", fetch_cpi),
                     ("GDP", fetch_gdp), ("OBR output gap", fetch_obr_output_gap)]:
        try:
            df = fn()
            print(f"{name}: {len(df)} rows saved")
        except Exception as e:  # keep going; the shipped snapshot stays in place
            print(f"{name}: download failed ({e}); keeping existing snapshot")
