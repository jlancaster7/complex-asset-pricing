#!/usr/bin/env python3
"""
Test script to visualize the regression relationship in Longstaff-Schwartz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.ls_pricing.instruments.bonds import CallableBond
from src.ls_pricing.core.curves import YieldCurve
from src.ls_pricing.core.hull_white import HullWhiteModel
from src.ls_pricing.engine.monte_carlo import MonteCarloEngine
from src.ls_pricing.engine.callable_bond import CallableBondEngine

# Load actual Treasury curve data
treasury_data = pd.read_csv("data/active_treasury_curve.csv", encoding="latin-1")


# Parse tenor strings to years
def tenor_to_years(tenor_str):
    if "M" in tenor_str and "Y" not in tenor_str:
        # Handle months
        months = int(tenor_str.replace("M", ""))
        return months / 12.0
    elif "W" in tenor_str:
        # Handle weeks
        weeks = int(tenor_str.replace("W", ""))
        return weeks / 52.0
    elif "Y" in tenor_str:
        # Handle years
        return float(tenor_str.replace("Y", ""))
    else:
        return None


# Extract and convert data
treasury_data["TenorYears"] = treasury_data["Tenor"].apply(tenor_to_years)
treasury_data["YieldDecimal"] = treasury_data["Yield"] / 100.0

# Filter out any rows with invalid tenor conversions and sort by tenor
valid_data = treasury_data.dropna(subset=["TenorYears"]).sort_values("TenorYears")

# Extract tenors and rates
tenors = valid_data["TenorYears"].values
rates = valid_data["YieldDecimal"].values

# Create yield curve
curve_date = datetime(2025, 7, 22)
yield_curve = YieldCurve(tenors, rates, curve_date)

# Create a callable bond
callable_bond = CallableBond(
    face_value=100.0,
    coupon_rate=0.055,
    payment_frequency=2,
    maturity=30.0,
    first_call_date=10.0,
    call_price=100.0,
    credit_spread=0.01,
)

# Setup Hull-White model
hw_model = HullWhiteModel(
    a=0.03, sigma=0.007, yield_curve=yield_curve  # Mean reversion  # Volatility
)

# Create Monte Carlo engine
mc_engine = MonteCarloEngine(
    model=hw_model, n_paths=2000, n_steps=60  # More paths for better visualization
)

# Create callable bond engine
engine = CallableBondEngine(mc_engine)

# Enable regression logging
engine.enable_regression_logging = True

print("Pricing callable bond with regression visualization...")
print("=" * 80)

result = engine.price_callable_bond(callable_bond)

print("\n" + "=" * 80)
print("FINAL RESULTS:")
print("=" * 80)
print(f"Callable bond price: ${result['callable_bond_price']:.2f}")
print(f"Straight bond price: ${result['straight_bond_price']:.2f}")
print(f"Option value: ${result['option_value']:.2f}")
print(f"Exercise probability: {result['exercise_probability']:.1%}")

print("\nKey insights from regression visualization:")
print("- Raw issuer values: Actual continuation values from backward induction")
print("- Smoothed values: Regression predictions that reduce noise")
print("- Exercise occurs when smoothed continuation > $100 (call price)")
print("- Lower rates → higher bond values → more likely to call")
