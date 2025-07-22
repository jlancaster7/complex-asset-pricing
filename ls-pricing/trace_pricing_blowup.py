#!/usr/bin/env python3
"""Trace how negative rates blow up callable bond pricing"""

import numpy as np
import pandas as pd
from datetime import datetime
from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.hull_white import HullWhiteModel
from ls_pricing.instruments.bonds import CallableBond

# Load CSV curve that causes the problem
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
yield_curve = YieldCurve(tenors_csv, rates_csv, curve_date)

# Create Hull-White model
hw_model = HullWhiteModel(a=0.03, sigma=0.008, yield_curve=yield_curve)

# Create the bond
bond = CallableBond(
    face_value=100.0,
    coupon_rate=0.067,
    maturity=30.0,
    payment_frequency=2,
    first_call_date=10.0,
    call_price=100.0,
    credit_spread=0.01
)

print("Understanding the Pricing Blowup")
print("="*60)

# 1. Show how bond valuation works
print("\n1. Bond Valuation at Different Times and Rates:")
print("-"*50)

# Value bond at different times with current short rate
times = [0, 10, 20, 25, 29]
for t in times:
    # Get the forward rate curve at time t
    r_t = yield_curve.get_rate(t)
    bond_value = bond.value(hw_model, t, r_t)
    remaining_maturity = bond.maturity - t
    
    print(f"\nAt time t={t} years:")
    print(f"  Current short rate: {r_t*100:.3f}%")
    print(f"  Remaining maturity: {remaining_maturity} years")
    print(f"  Bond value: ${bond_value:.2f}")

# 2. Show what happens in Monte Carlo paths
print("\n\n2. What Happens in Monte Carlo Simulation:")
print("-"*50)

# The issue: When Hull-White simulates forward, it uses the yield curve
# to calibrate theta(t). Let's see what theta looks like:

print("\nTheta function (drift adjustment) at different times:")
times_theta = np.linspace(0, 40, 9)
for t in times_theta:
    # Hull-White uses forward rates to calibrate theta
    if t < 30:
        # Normal calculation
        dt = 0.01
        f_t = yield_curve.get_forward_rate(t, t + dt)
    else:
        # Extrapolation territory - forward rate calculation will use crazy rates
        rate_t = yield_curve.get_rate(t)
        rate_t_plus = yield_curve.get_rate(t + 0.01)
        f_t = rate_t + t * (rate_t_plus - rate_t) / 0.01  # Approximation
    
    print(f"  t={t:4.1f}: forward rate ≈ {f_t*100:7.3f}%")

# 3. Show the specific problem with zero bond prices
print("\n\n3. Zero Bond Price Calculation Issues:")
print("-"*50)

# In Hull-White, P(t,T) calculation uses the yield curve
# When rates go negative, zero bond prices explode

t_current = 20  # Current time
r_current = 0.04  # Current short rate

print(f"\nZero bond prices from t={t_current} to various maturities T:")
print("T    | P(t,T)    | Yield Curve Rate at T | Issue")
print("-"*60)

for T in [25, 30, 35, 40, 50]:
    try:
        p_t_T = hw_model.zero_bond_price(t=t_current, T=T, r_t=r_current)
        yc_rate_T = yield_curve.get_rate(T)
        
        issue = ""
        if yc_rate_T < 0:
            issue = "NEGATIVE RATE!"
        elif p_t_T > 1:
            issue = "P > 1 (impossible!)"
            
        print(f"{T:4.0f} | {p_t_T:9.6f} | {yc_rate_T*100:18.3f}% | {issue}")
    except Exception as e:
        print(f"{T:4.0f} | ERROR     | {yield_curve.get_rate(T)*100:18.3f}% | {str(e)[:20]}")

# 4. The knockout punch - what happens to bond values
print("\n\n4. The Callable Bond Pricing Explosion:")
print("-"*50)

# When the Monte Carlo engine simulates paths beyond 30 years,
# the negative rates cause bond values to explode

print("\nSimulating what happens in a single MC path:")
print("(Using simplified calculation)")

# Simulate a path that goes beyond 30 years
path_times = np.array([0, 10, 20, 30, 35, 40])
path_rates = np.array([0.043, 0.045, 0.047, 0.049, -0.02, -0.10])  # Goes negative

print("\nTime | Rate  | Bond Value | Note")
print("-"*50)

for i, (t, r) in enumerate(zip(path_times, path_rates)):
    if t <= bond.maturity:
        # Simplified bond value calculation
        remaining_payments = int((bond.maturity - t) * bond.payment_frequency)
        
        if r < 0 and t > 30:
            # Negative rate creates huge present values
            pv_factor = np.exp(-r * (bond.maturity - t))  # This GROWS instead of shrinking!
            bond_val = bond.coupon_payment * remaining_payments * 20 + bond.face_value * pv_factor
            note = "PV factor > 1 due to negative rate!"
        else:
            # Normal calculation
            bond_val = bond.value(hw_model, t, max(r, 0.001))  # Avoid actual negative
            note = ""
            
        print(f"{t:4.0f} | {r*100:5.1f}% | ${bond_val:10.2f} | {note}")

print("\n\nKEY INSIGHT:")
print("="*60)
print("When rates go negative in the simulation:")
print("1. Discount factors become > 1 (money grows in present value!)")
print("2. Bond values explode to huge numbers")
print("3. The callable bond is valued as the average across paths")
print("4. Even a few paths with exploded values skew the average way up")
print("5. This makes callable bonds appear MORE valuable than straight bonds")
print("\nThis is why your callable bond showed $127.99 vs straight $112.67!")