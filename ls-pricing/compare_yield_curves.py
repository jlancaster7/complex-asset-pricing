#!/usr/bin/env python3
"""Compare hardcoded vs CSV yield curves"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
sys.path.append('./src')

from ls_pricing.core.curves import YieldCurve

# 1. Hardcoded yield curve from test
tenors_test = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
rates_test = np.array([0.043, 0.043, 0.041, 0.038, 0.038, 0.039, 0.041, 0.043, 0.046, 0.047])
curve_date = datetime(2025, 4, 20)
yield_curve_test = YieldCurve(tenors_test, rates_test, curve_date)

# 2. CSV-loaded yield curve from notebook
treasury_data = pd.read_csv('data/active_treasury_curve.csv', encoding='latin-1')

def tenor_to_years(tenor_str):
    if 'M' in tenor_str and 'Y' not in tenor_str:
        months = int(tenor_str.replace('M', ''))
        return months / 12.0
    elif 'W' in tenor_str:
        weeks = int(tenor_str.replace('W', ''))
        return weeks / 52.0
    elif 'Y' in tenor_str:
        return float(tenor_str.replace('Y', ''))
    else:
        return None

treasury_data['TenorYears'] = treasury_data['Tenor'].apply(tenor_to_years)
treasury_data['YieldDecimal'] = treasury_data['Yield'] / 100.0
valid_data = treasury_data.dropna(subset=['TenorYears']).sort_values('TenorYears')

tenors_csv = valid_data['TenorYears'].values
rates_csv = valid_data['YieldDecimal'].values
yield_curve_csv = YieldCurve(tenors_csv, rates_csv, curve_date)

# Compare the curves
print("Yield Curve Comparison")
print("="*60)
print("\nHardcoded (Test) Curve:")
print(f"Number of points: {len(tenors_test)}")
print(f"Tenors: {tenors_test}")
print(f"Rates: {rates_test * 100}")

print("\nCSV-Loaded (Notebook) Curve:")
print(f"Number of points: {len(tenors_csv)}")
print(f"Tenors: {tenors_csv}")
print(f"Rates: {rates_csv * 100}")

# Compare rates at key tenors
key_tenors = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
print("\nRate Comparison at Key Tenors:")
print("Tenor | Test Rate | CSV Rate | Difference (bps)")
print("-"*50)
for tenor in key_tenors:
    rate_test = yield_curve_test.get_rate(tenor)
    rate_csv = yield_curve_csv.get_rate(tenor)
    diff_bps = (rate_csv - rate_test) * 10000
    print(f"{tenor:5.1f} | {rate_test*100:9.3f}% | {rate_csv*100:8.3f}% | {diff_bps:+8.1f}")

# Check initial rate (important for pricing)
r0_test = yield_curve_test.get_rate(0.0)
r0_csv = yield_curve_csv.get_rate(0.0)
print(f"\nInitial rate (r0):")
print(f"Test: {r0_test*100:.3f}%")
print(f"CSV: {r0_csv*100:.3f}%")
print(f"Difference: {(r0_csv - r0_test)*10000:.1f} bps")