import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

from ls_pricing.core.curves import YieldCurve


def parse_tenor_to_years(tenor_str: str) -> float | None:
    """Convert tenor strings like '6M', '2Y', '13W' to years.
    Returns None if cannot parse.
    """
    if tenor_str is None or not isinstance(tenor_str, str):
        return None
    s = tenor_str.strip().upper()
    if "M" in s and "Y" not in s and "W" not in s:
        try:
            months = int(s.replace("M", ""))
            return months / 12.0
        except Exception:
            return None
    if "W" in s and "Y" not in s:
        try:
            weeks = int(s.replace("W", ""))
            return weeks / 52.0
        except Exception:
            return None
    if "Y" in s:
        try:
            years = float(s.replace("Y", ""))
            return years
        except Exception:
            return None
    return None


def load_yield_curve_from_csv(
    csv_path: str,
    curve_date: datetime,
    tenor_col: str = "Tenor",
    yield_col: str = "Yield",
    encoding: str = "latin-1",
) -> YieldCurve:
    """Load a yield curve from a CSV with tenor strings and yield in percent.
    - tenor_col contains strings like '6M', '2Y', '13W'.
    - yield_col contains numeric yields in percent.
    Returns a YieldCurve with tenors (years) and rates (decimal).
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    df["TenorYears"] = df[tenor_col].apply(parse_tenor_to_years)
    df["YieldDecimal"] = df[yield_col] / 100.0
    valid = df.dropna(subset=["TenorYears"]).sort_values("TenorYears")
    tenors = valid["TenorYears"].to_numpy()
    rates = valid["YieldDecimal"].to_numpy()
    return YieldCurve(tenors, rates, curve_date)
