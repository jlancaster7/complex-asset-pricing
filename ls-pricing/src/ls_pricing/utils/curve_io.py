import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional

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


def load_mmd_par_yield_curve_from_csv(
    csv_path: str,
    rating_type: str = "AAA",
    curve_is_callable: str = "YES",
    business_date: Optional[datetime] = None,
    encoding: str = "latin-1",
    coupon_frequency: int = 1,
) -> YieldCurve:
    """Load an MMD municipal par-yield curve and bootstrap a zero curve.

    Assumptions/Notes:
    - Input columns are like '1Y_MMD', '2Y_MMD', ..., in PERCENT yields.
    - We treat provided values as PAR YIELDS with the given coupon_frequency.
    - If coupon_frequency=1 (annual), bootstrap is exact with one DF per year.
    - For muni accuracy, semiannual coupons (2) are typical, but annual bootstrap
      is used here to avoid underdetermined system given only annual par points.
    - Returns a YieldCurve with zero/spot rates (continuous comp) at year nodes.

    Parameters:
    - rating_type: e.g., 'AAA', 'AA', 'A', 'BAA'.
    - curve_is_callable: 'YES' or 'NO' selection from the file.
    - business_date: if None, latest date in the file will be selected.
    - coupon_frequency: 1 (annual) or 2 (semiannual). Only 1 is exact given data.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    # Normalize headers: strip BOM variants then lowercase
    def _norm_col(c: object) -> str:
        s = str(c)
        # Remove true BOM and common mis-decoded sequence
        s = s.replace("\ufeff", "").replace("ï»¿", "").strip().lower()
        return s
    df.columns = [_norm_col(c) for c in df.columns]

    # Filter by rating and callable flag (case-insensitive arguments)
    mask = (
        df["rating_type"].str.upper() == rating_type.upper()
    ) & (df["curve_is_callable"].str.upper() == curve_is_callable.upper())
    df_f = df.loc[mask].copy()
    if df_f.empty:
        raise ValueError(
            f"No rows found for rating_type={rating_type} and curve_is_callable={curve_is_callable}"
        )

    def _parse_bd(s: str) -> datetime:
        return datetime.strptime(str(s), "%m/%d/%Y") if "/" in str(s) else datetime.fromisoformat(str(s))

    df_f["_bd"] = df_f["business_date"].apply(_parse_bd)
    if business_date is not None:
        df_f = df_f.sort_values("_bd")
        chosen = df_f.loc[df_f["_bd"] == business_date]
        if chosen.empty:
            chosen = df_f.loc[df_f["_bd"] <= business_date].tail(1)
            if chosen.empty:
                chosen = df_f.head(1)
        row = chosen.iloc[0]
    else:
        row = df_f.loc[df_f["_bd"].idxmax()]

    # Robust Python datetime
    bd_val = row["_bd"]
    if isinstance(bd_val, datetime):
        curve_date = bd_val
    else:
        try:
            curve_date = datetime.strptime(str(bd_val), "%Y-%m-%d %H:%M:%S")
        except Exception:
            try:
                curve_date = datetime.strptime(str(bd_val), "%Y-%m-%d")
            except Exception:
                curve_date = _parse_bd(str(bd_val))

    # Extract year-tenor par yields (columns like '1y_mmd', '2y_mmd', ...)
    par_cols = [c for c in df.columns if c.endswith("_mmd")]
    tenors_years: list[int] = []
    par_yields: list[float] = []
    for c in par_cols:
        try:
            # '10y_mmd' -> split on 'y_'
            if "y_" in c:
                year_part = c.split("y_")[0]
            else:
                year_part = c.replace("_mmd", "").rstrip("y")
            y = int(year_part)
            val_raw = row.at[c] if c in row.index else None
            if val_raw is None:
                continue
            try:
                val = float(val_raw)
            except (TypeError, ValueError):
                continue
            tenors_years.append(y)
            par_yields.append(val / 100.0)
        except Exception:
            continue

    pairs = sorted(zip(tenors_years, par_yields), key=lambda x: x[0])
    tenors_years = [p[0] for p in pairs]
    par_yields = [p[1] for p in pairs]

    if len(tenors_years) == 0:
        raise ValueError("No tenor columns like '1Y_MMD' found in the selected row.")

    if coupon_frequency != 1:
        import warnings
        warnings.warn(
            "Semiannual bootstrap not supported with only annual par points; using annual coupon bootstrap.")
        coupon_frequency = 1

    # Annual-coupon bootstrap at integer years
    dfs = []
    for n_years, y_par in zip(tenors_years, par_yields):
        sum_prior = 0.0
        for k, df_k in zip(tenors_years[: len(dfs)], dfs):
            if k < n_years:
                sum_prior += y_par * df_k
        df_n = (1.0 - sum_prior) / (1.0 + y_par)
        dfs.append(df_n)

    tenors = np.array(tenors_years, dtype=float)
    dfs_arr = np.array(dfs, dtype=float)
    if np.any(dfs_arr <= 0):
        raise ValueError("Bootstrapped discount factors contain non-positive values; check input data.")
    zero_rates = -np.log(dfs_arr) / tenors

    return YieldCurve(tenors=tenors, rates=zero_rates, curve_date=curve_date)
