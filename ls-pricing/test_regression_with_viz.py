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
from ls_pricing.utils.curve_io import load_yield_curve_from_csv

# Load actual Treasury curve data -> yield curve
curve_date = datetime(2025, 7, 22)
yield_curve = load_yield_curve_from_csv(
    "data/active_treasury_curve.csv",
    curve_date,
    tenor_col="Tenor",
    yield_col="Yield",
    encoding="latin-1",
)

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
