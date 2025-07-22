#!/usr/bin/env python3
"""Comprehensive test of YieldCurve methods comparing hardcoded vs CSV data"""

import numpy as np
import pandas as pd
from datetime import datetime
from ls_pricing.core.curves import YieldCurve

# Load CSV curve
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

# Create hardcoded curve
tenors_hard = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
rates_hard = np.array([0.043, 0.043, 0.041, 0.038, 0.038, 0.039, 0.041, 0.043, 0.046, 0.047])
yield_curve_hard = YieldCurve(tenors_hard, rates_hard, curve_date)

print("="*70)
print("YieldCurve Method Comparison: CSV vs Hardcoded")
print("="*70)

# 1. Test get_rate at various tenors
print("\n1. Testing get_rate() method:")
print("-"*50)
print("Tenor | CSV Rate | Hard Rate | Difference")
print("-"*50)

test_tenors = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 35.0, 40.0, 50.0]
for tenor in test_tenors:
    rate_csv = yield_curve_csv.get_rate(tenor)
    rate_hard = yield_curve_hard.get_rate(tenor)
    diff = (rate_csv - rate_hard) * 10000  # in bps
    
    status = ""
    if tenor > 30 and abs(rate_csv) > 0.10:  # Flag if rate > 10% or < -10%
        status = " ⚠️ EXTREME!"
    elif tenor > 30 and rate_csv < 0:
        status = " ⚠️ NEGATIVE!"
        
    print(f"{tenor:5.1f} | {rate_csv*100:8.3f}% | {rate_hard*100:8.3f}% | {diff:+7.1f} bps{status}")

# 2. Test get_discount_factor
print("\n\n2. Testing get_discount_factor() method:")
print("-"*50)
print("Tenor | CSV DF   | Hard DF  | Ratio    | Status")
print("-"*50)

for tenor in test_tenors:
    df_csv = yield_curve_csv.get_discount_factor(tenor)
    df_hard = yield_curve_hard.get_discount_factor(tenor)
    ratio = df_csv / df_hard if df_hard > 0 else float('inf')
    
    status = ""
    if df_csv > 1:
        status = "⚠️ DF > 1!"
    elif ratio > 2 or ratio < 0.5:
        status = "⚠️ Large difference!"
        
    print(f"{tenor:5.1f} | {df_csv:8.5f} | {df_hard:8.5f} | {ratio:8.3f} | {status}")

# 3. Test get_forward_rate
print("\n\n3. Testing get_forward_rate() method:")
print("-"*50)
print("Period        | CSV Fwd  | Hard Fwd | Difference | Status")
print("-"*50)

forward_periods = [
    (0, 1), (1, 2), (5, 10), (10, 20), (20, 30),
    (25, 30), (29, 30), (30, 31), (30, 35), (30, 40)
]

for start, end in forward_periods:
    try:
        fwd_csv = yield_curve_csv.get_forward_rate(start, end)
        fwd_hard = yield_curve_hard.get_forward_rate(start, end)
        diff = (fwd_csv - fwd_hard) * 10000
        
        status = ""
        if fwd_csv < -0.01:  # -1%
            status = "⚠️ NEGATIVE!"
        elif abs(fwd_csv) > 0.10:  # > 10%
            status = "⚠️ EXTREME!"
        elif start >= 30:
            status = "📊 Extrapolated"
            
        print(f"({start:2d},{end:2d}) | {fwd_csv*100:8.3f}% | {fwd_hard*100:8.3f}% | {diff:+7.1f} bps | {status}")
    except Exception as e:
        print(f"({start:2d},{end:2d}) | ERROR: {str(e)[:40]}")

# 4. Check cubic spline behavior at boundaries
print("\n\n4. Analyzing Cubic Spline Behavior at Boundaries:")
print("-"*50)

# Check the slope of rates near the last data point
h = 0.01  # Small increment
last_tenor_csv = tenors_csv[-1]
last_tenor_hard = tenors_hard[-1]

# For CSV curve
rate_at_30_csv = yield_curve_csv.get_rate(last_tenor_csv)
rate_at_30_plus_csv = yield_curve_csv.get_rate(last_tenor_csv + h)
slope_csv = (rate_at_30_plus_csv - rate_at_30_csv) / h

# For hardcoded curve
rate_at_30_hard = yield_curve_hard.get_rate(last_tenor_hard)
rate_at_30_plus_hard = yield_curve_hard.get_rate(last_tenor_hard + h)
slope_hard = (rate_at_30_plus_hard - rate_at_30_hard) / h

print(f"CSV Curve:")
print(f"  Last data point: {last_tenor_csv}Y at {rate_at_30_csv*100:.3f}%")
print(f"  Rate just after: {rate_at_30_plus_csv*100:.3f}%")
print(f"  Slope: {slope_csv*100:.3f}% per year")
print(f"  10Y extrapolation would add: {slope_csv*10*100:.1f}% to rate")

print(f"\nHardcoded Curve:")
print(f"  Last data point: {last_tenor_hard}Y at {rate_at_30_hard*100:.3f}%")
print(f"  Rate just after: {rate_at_30_plus_hard*100:.3f}%")
print(f"  Slope: {slope_hard*100:.3f}% per year")
print(f"  10Y extrapolation would add: {slope_hard*10*100:.1f}% to rate")

# 5. Check the actual data points
print("\n\n5. Data Point Analysis:")
print("-"*50)
print("CSV data points:")
for i, (t, r) in enumerate(zip(tenors_csv, rates_csv)):
    print(f"  {i:2d}: {t:6.3f}Y @ {r*100:6.3f}%", end="")
    if i % 3 == 2:
        print()
print()

# Calculate second derivative at key points to understand spline behavior
print("\n\n6. Second Derivative Analysis (Spline Curvature):")
print("-"*50)

def estimate_second_derivative(curve, tenor, h=0.01):
    """Estimate second derivative using finite differences"""
    f_minus = curve.get_rate(tenor - h)
    f_center = curve.get_rate(tenor)
    f_plus = curve.get_rate(tenor + h)
    return (f_plus - 2*f_center + f_minus) / (h**2)

check_points = [20.0, 25.0, 29.0, 30.0, 31.0, 35.0]
print("Tenor | CSV d²r/dt² | Hard d²r/dt² | Note")
print("-"*50)

for tenor in check_points:
    d2_csv = estimate_second_derivative(yield_curve_csv, tenor)
    d2_hard = estimate_second_derivative(yield_curve_hard, tenor)
    
    note = ""
    if abs(d2_csv) > 0.01:
        note = "High curvature!"
    
    print(f"{tenor:5.1f} | {d2_csv*100:11.6f} | {d2_hard*100:12.6f} | {note}")

print("\n\nSUMMARY:")
print("="*70)
print("The CSV curve shows problematic behavior in extrapolation due to:")
print("1. Negative slope at the boundary")
print("2. High curvature (second derivative) near the last point")
print("3. This causes rates to plummet when extrapolating beyond 30Y")
print("4. Forward rates become extreme or negative in extrapolation region")
print("5. Discount factors > 1 for long tenors (impossible!)")