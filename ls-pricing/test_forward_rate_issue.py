#!/usr/bin/env python3
"""Deep dive into forward rate calculation issues"""

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

print("Forward Rate Calculation Deep Dive")
print("="*60)

# Let's trace through the forward rate calculation step by step
print("\n1. Understanding Forward Rate Calculation:")
print("   Forward rate f(t1,t2) = -ln(P(0,t2)/P(0,t1)) / (t2-t1)")
print("   Where P(0,t) = exp(-r(t) * t)")

# Test case: Forward rate from 30 to 31 years
t1, t2 = 30.0, 31.0
print(f"\n2. Calculating forward rate from {t1}Y to {t2}Y:")
print("-"*50)

# Get spot rates
r_t1 = yield_curve_csv.get_rate(t1)
r_t2 = yield_curve_csv.get_rate(t2)
print(f"Spot rate at {t1}Y: {r_t1*100:.3f}%")
print(f"Spot rate at {t2}Y: {r_t2*100:.3f}%")

# Get discount factors
df_t1 = yield_curve_csv.get_discount_factor(t1)
df_t2 = yield_curve_csv.get_discount_factor(t2)
print(f"\nDiscount factor P(0,{t1}): {df_t1:.6f}")
print(f"Discount factor P(0,{t2}): {df_t2:.6f}")
print(f"Ratio P(0,{t2})/P(0,{t1}): {df_t2/df_t1:.6f}")

# Calculate forward rate manually
forward_manual = -np.log(df_t2 / df_t1) / (t2 - t1)
forward_method = yield_curve_csv.get_forward_rate(t1, t2)
print(f"\nForward rate (manual calc): {forward_manual*100:.3f}%")
print(f"Forward rate (method call): {forward_method*100:.3f}%")
print(f"Difference: {abs(forward_manual - forward_method)*10000:.1f} bps")

# Show what's happening with instantaneous forward rates
print("\n3. Instantaneous Forward Rates (used in Hull-White theta):")
print("-"*50)
print("Time | Spot Rate | Inst. Forward | Note")
print("-"*50)

dt = 0.001  # Very small increment for instantaneous rate
test_times = [0, 5, 10, 20, 25, 29, 30, 31, 35, 40]

for t in test_times:
    try:
        spot = yield_curve_csv.get_rate(t)
        
        # Instantaneous forward rate approximation
        if t < 40:  # Avoid going too far into extrapolation
            inst_fwd = yield_curve_csv.get_forward_rate(t, t + dt)
        else:
            inst_fwd = float('nan')
        
        note = ""
        if t > 30:
            note = "Extrapolated"
        if inst_fwd < 0:
            note += " NEGATIVE!"
            
        print(f"{t:4.0f} | {spot*100:9.3f}% | {inst_fwd*100:13.3f}% | {note}")
    except Exception as e:
        print(f"{t:4.0f} | ERROR: {str(e)}")

# Show Hull-White theta calculation
print("\n4. Hull-White Theta Function (drift adjustment):")
print("-"*50)
print("This is where the problem propagates to Monte Carlo simulation")

# Simplified theta calculation (from hull_white.py _theta method)
print("\nTime | Theta(t) | Note")
print("-"*40)

a = 0.03  # Hull-White mean reversion
for t in [0, 10, 20, 25, 29, 30, 31, 35]:
    try:
        dt = 1e-6
        f_t = yield_curve_csv.get_forward_rate(t, t + dt)
        f_t_plus = yield_curve_csv.get_forward_rate(t + dt, t + 2*dt)
        df_dt = (f_t_plus - f_t) / dt
        
        theta = df_dt + a * f_t + 0.5 * 0.008**2 * (1 - np.exp(-2*a*t)) / a
        
        note = ""
        if t > 30:
            note = "Extrapolated"
        if theta < -0.1 or theta > 0.1:
            note += " EXTREME!"
            
        print(f"{t:4.0f} | {theta*100:8.3f}% | {note}")
    except Exception as e:
        print(f"{t:4.0f} | ERROR")

print("\n5. Why This Breaks Monte Carlo:")
print("-"*50)
print("When theta becomes extreme or negative:")
print("- The drift term in dr = [theta - a*r]dt + sigma*dW goes haywire")
print("- Simulated rates can become negative or explode")
print("- Negative rates → discount factors > 1 → bond values explode")
print("- This is why callable bond prices higher than straight!")

# Compare with a well-behaved curve
print("\n6. Contrast with Hardcoded Curve:")
print("-"*50)
tenors_hard = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
rates_hard = np.array([0.043, 0.043, 0.041, 0.038, 0.038, 0.039, 0.041, 0.043, 0.046, 0.047])
yield_curve_hard = YieldCurve(tenors_hard, rates_hard, curve_date)

print("30Y → 31Y forward rates:")
fwd_csv = yield_curve_csv.get_forward_rate(30, 31)
fwd_hard = yield_curve_hard.get_forward_rate(30, 31)
print(f"CSV curve: {fwd_csv*100:.3f}%")
print(f"Hardcoded: {fwd_hard*100:.3f}%")
print(f"Difference: {(fwd_csv - fwd_hard)*10000:.0f} bps!")

print("\nTHE ROOT CAUSE:")
print("="*60)
print("The CSV data has many short-term points (1M, 6W, 2M, etc.)")
print("CubicSpline tries to fit all these points smoothly")
print("This creates unwanted curvature at the long end")
print("The negative slope at 30Y causes catastrophic extrapolation")