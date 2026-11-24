"""Sanity tests. Run with:  python -m pytest -q"""
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dataset  # noqa: E402
import fetch_data  # noqa: E402
from taylor import INERTIAL, TAYLOR_1993, TAYLOR_1999, taylor_rate  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return dataset.build_panel(save=False)


def test_panel_complete_and_monthly(panel):
    assert panel.index[0] == pd.Period("1997-05", "M")
    assert panel.index.is_monotonic_increasing and panel.index.is_unique
    assert len(panel) == len(pd.period_range(panel.index[0], panel.index[-1], freq="M"))
    assert panel.notna().all().all()


def test_known_data_points(panel):
    # Checked against BoE IUDBEDR and ONS D7G7 / DKO8
    assert panel.loc["1997-05", "bank_rate"] == 6.25
    assert panel.loc["2009-03", "bank_rate"] == 0.5
    assert panel.loc["2020-03", "bank_rate"] == 0.1
    assert panel.loc["2022-10", "bank_rate"] == 2.25   # 2.25% from 22 Sep, 3% from 3 Nov 2022
    assert panel.loc["2023-08", "bank_rate"] == 5.25
    assert panel.loc["2022-10", "cpi"] == 11.1          # UK CPI peak
    assert panel.loc["2023-05", "core_cpi"] == 7.1
    assert panel.loc["2022-10", "gap_obr"] == pytest.approx(1.03)  # OBR FY2022-23 placed at Oct


def test_headline_by_hand(panel):
    i = taylor_rate(panel["cpi"], panel["gap_obr"], 2.0, TAYLOR_1993)["2022-10"]
    assert i == pytest.approx(2.0 + 11.1 + 0.5 * (11.1 - 2.0) + 0.5 * 1.03)
    assert round(i, 1) == 18.2 and round(i - 2.25, 1) == 15.9


def test_rule_properties():
    pi, gap = pd.Series([2.0, 3.0]), pd.Series([0.0, 0.0])
    # at target with closed gap the rule returns the nominal neutral rate
    assert taylor_rate(pi, gap, 1.0)[0] == 3.0
    # Taylor principle: +1pp inflation -> +1.5pp rate
    assert taylor_rate(pi, gap, 1.0).diff()[1] == 1.5
    assert taylor_rate(pd.Series([2.0]), pd.Series([1.0]), 0.0, TAYLOR_1999)[0] == 3.0
    with pytest.raises(ValueError):
        taylor_rate(pi, gap, 1.0, INERTIAL)


def test_hp_filter():
    t = np.arange(60, dtype=float)
    assert np.allclose(dataset.hp_filter(3 + 0.5 * t), 3 + 0.5 * t)  # linear trend passes through
    rng = np.random.default_rng(0)
    y = 0.5 * t + rng.normal(size=60)
    assert abs((y - dataset.hp_filter(y)).mean()) < 1e-8  # cycle averages to zero
    rt = dataset.hp_one_sided(y, min_obs=40)
    assert np.isnan(rt[:39]).all() and rt[-1] == pytest.approx(dataset.hp_filter(y)[-1])


def _resp(text=None, js=None):
    r = mock.Mock()
    r.text, r.json.return_value = text, js
    r.raise_for_status.return_value = None
    return r


def test_fetch_parsers(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "RAW", tmp_path)
    boe = "DATE,IUDBEDR\n21 Sep 2022,1.75\n22 Sep 2022,2.25\n31 Oct 2022,2.25\n03 Nov 2022,3\n"
    with mock.patch.object(fetch_data, "_get", return_value=_resp(text=boe)):
        br = fetch_data.fetch_bank_rate()
    assert br.to_dict("list") == {"month": ["2022-09", "2022-10", "2022-11"], "bank_rate": [2.25, 2.25, 3.0]}

    ons = {"months": [{"date": "2022 OCT", "year": "2022", "value": "11.1"},
                      {"date": "2022 NOV", "year": "2022", "value": "10.7"}]}
    with mock.patch.object(fetch_data, "_get", return_value=_resp(js=ons)):
        cpi = fetch_data.fetch_cpi()
    assert list(cpi.index) == ["2022-10", "2022-11"] and cpi.iloc[0, 0] == 11.1


def test_results_match_readme():
    root = Path(__file__).resolve().parents[1]
    res = (root / "results" / "results.md").read_text()
    readme = (root / "README.md").read_text()
    for s in ["18.2%", "2.25%", "15.9pp", "October 2022"]:
        assert s in res
    for s in ["18.2%", "2.25%", "15.9", "October 2022"]:
        assert s in readme
