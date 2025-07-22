#!/usr/bin/env python3
"""Reproduce the exact notebook cell 8 results"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys

sys.path.append("./src")

from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.hull_white import HullWhiteModel
from ls_pricing.engine.monte_carlo import MonteCarloEngine
from ls_pricing.engine.callable_bond import CallableBondEngine
from ls_pricing.instruments.bonds import CallableBond

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

# Setup Hull-White model
a = 0.03  # Lower mean reversion for munis
sigma = 0.008  # Lower volatility for munis

hw_model = HullWhiteModel(a=a, sigma=sigma, yield_curve=yield_curve)

# Create Monte Carlo engine
mc_engine = MonteCarloEngine(
    model=hw_model,
    n_paths=5000,  # More paths for better convergence
    n_steps=60,  # Semi-annual steps for 30 years
)

# Create callable bond pricing engine
cb_engine = CallableBondEngine(
    mc_engine=mc_engine, basis_type="laguerre", basis_degree=3
)

# Create the callable municipal bond with credit spread
muni_bond = CallableBond(
    face_value=100.0,
    coupon_rate=0.067,  # 6.7% annual coupon
    maturity=30.0,  # 30 year maturity
    payment_frequency=2,  # Semi-annual payments
    first_call_date=10.0,  # Callable after 10 years
    call_price=100.0,  # Callable at par
    credit_spread=0.01,  # 100bp spread over risk-free rate
)

# Price the callable bond
print("Pricing callable municipal bond...")
# Pre-generate paths with fixed seed
cb_engine.generate_paths(30.0, seed=42)
result = cb_engine.price_callable_bond(muni_bond, from_investor_perspective=True)

# Display results
print("\nPricing Results:")
print("=" * 50)
print(f"Callable Bond Price: ${result['callable_bond_price']:.2f}")
print(f"Straight (Non-Callable) Bond Price: ${result['straight_bond_price']:.2f}")
print(f"Embedded Call Option Value: ${result['option_value']:.2f}")
print(f"\nCall Statistics:")
print(f"  Probability of Call: {result['exercise_probability']*100:.1f}%")
if result["mean_call_time"] > 0:
    print(f"  Expected Call Time (if called): {result['mean_call_time']:.1f} years")
print(f"\nYield Analysis:")
ytc = muni_bond.yield_to_maturity(result["callable_bond_price"])
print(f"  Yield to Maturity (if not called): {ytc*100:.2f}%")

# Calculate option-adjusted spread (OAS)
# This is a simplified calculation
avg_curve_yield = np.mean(rates)
oas = ytc - avg_curve_yield
print(f"  Option-Adjusted Spread (approx): {oas*10000:.0f} bps")
