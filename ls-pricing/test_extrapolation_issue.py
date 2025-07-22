#!/usr/bin/env python3
"""Demonstrate YieldCurve extrapolation issue with CSV vs hardcoded data"""

import numpy as np
import pandas as pd
from datetime import datetime
from ls_pricing.core.curves import YieldCurve

# 1. Load CSV curve
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
curve_date = datetime(2025, 4, 20)
yield_curve_csv = YieldCurve(tenors_csv, rates_csv, curve_date)

# 2. Hardcoded curve
tenors_hard = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
rates_hard = np.array([0.043, 0.043, 0.041, 0.038, 0.038, 0.039, 0.041, 0.043, 0.046, 0.047])
yield_curve_hard = YieldCurve(tenors_hard, rates_hard, curve_date)

print("CSV Curve Data Points:")
print(f"Tenors: {tenors_csv}")
print(f"Rates: {rates_csv * 100}")
print(f"\nFirst tenor: {tenors_csv[0]:.4f}")
print(f"Last tenor: {tenors_csv[-1]:.4f}")

print("\n" + "="*60)
print("Extrapolation Behavior Comparison")
print("="*60)

# Test various tenors including extrapolation
test_tenors = [0.0, 0.1, 1.0, 10.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

print("\nTenor | CSV Rate | Hard Rate | CSV-Hard Diff")
print("-"*50)
for t in test_tenors:
    rate_csv = yield_curve_csv.get_rate(t)
    rate_hard = yield_curve_hard.get_rate(t)
    diff = rate_csv - rate_hard
    print(f"{t:5.1f} | {rate_csv*100:9.3f}% | {rate_hard*100:9.3f}% | {diff*100:+9.3f}%")

# Show the problem with discount factors
print("\n" + "="*60)
print("Discount Factor Issues (CSV curve)")
print("="*60)
print("\nTenor | Discount Factor | Implied Rate")
print("-"*40)
for t in [30, 40, 50, 60, 70, 80, 90, 100]:
    df = yield_curve_csv.get_discount_factor(t)
    implied_rate = -np.log(df) / t if df > 0 else float('inf')
    print(f"{t:5.0f} | {df:15.6f} | {implied_rate*100:9.3f}%")

# Let's check the slope at the boundary
print("\n" + "="*60)
print("Cubic Spline Behavior at Boundaries")
print("="*60)

# Check derivative at last point
last_tenor = tenors_csv[-1]
h = 0.001
rate_at_last = yield_curve_csv.get_rate(last_tenor)
rate_just_after = yield_curve_csv.get_rate(last_tenor + h)
slope_at_boundary = (rate_just_after - rate_at_last) / h

print(f"\nCSV curve:")
print(f"Last data point: tenor={last_tenor:.1f}, rate={rate_at_last*100:.3f}%")
print(f"Slope at boundary: {slope_at_boundary*100:.3f}% per year")
print(f"This means rate changes by {slope_at_boundary*10*100:.1f}% over 10 years!")

# Check for hardcoded curve too
last_tenor_hard = tenors_hard[-1]
rate_at_last_hard = yield_curve_hard.get_rate(last_tenor_hard)
rate_just_after_hard = yield_curve_hard.get_rate(last_tenor_hard + h)
slope_at_boundary_hard = (rate_just_after_hard - rate_at_last_hard) / h

print(f"\nHardcoded curve:")
print(f"Last data point: tenor={last_tenor_hard:.1f}, rate={rate_at_last_hard*100:.3f}%")
print(f"Slope at boundary: {slope_at_boundary_hard*100:.3f}% per year")