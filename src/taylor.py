"""Taylor-rule variants.

All rules are of the form  i* = r* + pi + a_pi (pi - pi*) + a_y * gap
(the 'pi' term converts the real neutral rate r* into a nominal rate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    a_pi: float = 0.5      # response to inflation deviation (Taylor 1993: 0.5 -> total 1.5 on pi)
    a_y: float = 0.5       # response to output gap (Taylor 1993: 0.5; Taylor 1999 / 'balanced': 1.0)
    rho: float = 0.0       # interest-rate smoothing per month (0 = no inertia)
    floor: float | None = None  # effective lower bound, e.g. 0.1


TAYLOR_1993 = Rule("Taylor (1993)")
TAYLOR_1999 = Rule("Taylor (1999), a_y = 1", a_y=1.0)
INERTIAL = Rule("Inertial Taylor (ρ = 0.85/qtr)", rho=0.85 ** (1 / 3))


def taylor_rate(pi: pd.Series, gap: pd.Series, r_star: float, rule: Rule = TAYLOR_1993,
                pi_target: float = 2.0, actual: pd.Series | None = None) -> pd.Series:
    """Rule-implied Bank Rate (per cent).

    With rule.rho > 0 the rate is smoothed:  i_t = rho * i_{t-1} + (1 - rho) * i*_t,
    where i_{t-1} is the *actual* Bank Rate last month (as in Clarida-Gali-Gertler), so
    the inertial rule answers 'where would the MPC go next, given where it is now?'.
    """
    target = r_star + pi + rule.a_pi * (pi - pi_target) + rule.a_y * gap
    if rule.rho:
        if actual is None:
            raise ValueError("inertial rule needs the actual rate path")
        target = rule.rho * actual.shift(1) + (1 - rule.rho) * target
    if rule.floor is not None:
        target = np.maximum(target, rule.floor)
    return target.rename(rule.name)
