#!/usr/bin/env python3
"""Debug the issuer value calculations in callable bond pricing"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
sys.path.append('./src')

from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.hull_white import HullWhiteModel
from ls_pricing.engine.monte_carlo import MonteCarloEngine
from ls_pricing.engine.callable_bond import CallableBondEngine
from ls_pricing.instruments.bonds import CallableBond

# Create a simple yield curve
tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
rates = np.array([0.043, 0.043, 0.041, 0.038, 0.038, 0.039, 0.041, 0.043, 0.046, 0.047])
curve_date = datetime(2025, 4, 20)
yield_curve = YieldCurve(tenors, rates, curve_date)

# Model parameters
a = 0.03
sigma = 0.008
hw_model = HullWhiteModel(a=a, sigma=sigma, yield_curve=yield_curve)

# Create MC engine with fewer paths for debugging
mc_engine = MonteCarloEngine(model=hw_model, n_paths=100, n_steps=60)

# Create callable bond pricing engine
cb_engine = CallableBondEngine(mc_engine=mc_engine, basis_type='laguerre', basis_degree=3)

# Create callable bond with 6.7% coupon
bond = CallableBond(
    face_value=100.0,
    coupon_rate=0.067,
    maturity=30.0,
    payment_frequency=2,
    first_call_date=10.0,
    call_price=100.0,
    credit_spread=0.01
)

# Generate paths with fixed seed
print("Generating paths...")
cb_engine.generate_paths(30.0, seed=42)

# Access the engine's cached paths
paths = cb_engine._cached_paths
rates = paths["rates"]
times = paths["times"]

# Get exercise times (only after first call date)
exercise_indices = bond.get_exercise_times(times)
print(f"\nNumber of exercise opportunities: {len(exercise_indices)}")
print(f"First exercise index: {exercise_indices[0] if exercise_indices else 'None'}")
print(f"First exercise time: {times[exercise_indices[0]] if exercise_indices else 'None'}")

# Let's manually trace through the backward induction
n_paths = rates.shape[0]
issuer_values = np.zeros((n_paths, len(times)))

# At maturity, issuer owes final payment
terminal_idx = len(times) - 1
final_payment = bond.coupon_payment + bond.face_value
issuer_values[:, terminal_idx] = -final_payment
print(f"\nAt maturity (t={times[terminal_idx]:.1f}), issuer owes: ${final_payment:.2f}")

# Backward induction through ALL time steps
print("\nBackward induction (showing first 5 paths):")
print("="*60)

for step in range(len(times) - 2, -1, -1):
    current_time = times[step]
    next_step = step + 1
    
    # Discount issuer values from next period
    dt = times[next_step] - times[step]
    discount_rates = rates[:, step] + bond.credit_spread
    discount_factors = np.exp(-discount_rates * dt)
    issuer_values[:, step] = issuer_values[:, next_step] * discount_factors
    
    # Add coupon payment if this is a coupon date
    if cb_engine._is_coupon_date(current_time, bond):
        issuer_values[:, step] -= bond.coupon_payment
        
    # Show values for first few paths at key time points
    if step in [0, exercise_indices[0] if exercise_indices else -1, 30, 59]:
        print(f"\nStep {step}, Time {current_time:.2f} years:")
        print("Path | Rate | Issuer Value | Discount Factor")
        for i in range(min(5, n_paths)):
            print(f"{i:4d} | {rates[i,step]*100:5.2f}% | ${issuer_values[i,step]:10.2f} | {discount_factors[i]:.4f}")

# Now check the exercise decisions at first call date
if exercise_indices:
    first_call_idx = exercise_indices[0]
    print(f"\n\nExercise Analysis at First Call Date (t={times[first_call_idx]:.1f}):")
    print("="*60)
    
    # Calculate bond values at first call
    bond_values = bond.value_at_node(
        hw_model, 
        times[first_call_idx], 
        rates[:, first_call_idx]
    )
    
    # Immediate exercise value
    immediate_ex_value = bond_values - bond.call_price
    
    # Continuation value (issuer's future liability)
    continuation = issuer_values[:, first_call_idx].copy()
    
    # Exercise decision
    immediate_issuer_value = -bond.call_price
    exercise = (immediate_issuer_value > continuation) & (immediate_ex_value > 0)
    
    print(f"\nFirst 10 paths at first call date:")
    print("Path | Rate | Bond Val | Imm Ex Val | Continuation | Call?")
    for i in range(min(10, n_paths)):
        print(f"{i:4d} | {rates[i,first_call_idx]*100:5.2f}% | ${bond_values[i]:7.2f} | "
              f"${immediate_ex_value[i]:8.2f} | ${continuation[i]:11.2f} | {'Yes' if exercise[i] else 'No'}")
    
    print(f"\nStatistics:")
    print(f"Mean bond value: ${bond_values.mean():.2f}")
    print(f"Bonds above par: {(bond_values > 100).sum()} / {n_paths}")
    print(f"Bonds called: {exercise.sum()} / {n_paths}")
    print(f"Mean continuation value: ${continuation.mean():.2f}")
    print(f"Immediate issuer value if called: ${immediate_issuer_value:.2f}")

# Final pricing
callable_price = -np.mean(issuer_values[:, 0])
print(f"\n\nFinal Results:")
print("="*60)
print(f"Mean issuer value at t=0: ${np.mean(issuer_values[:, 0]):.2f}")
print(f"Callable bond price: ${callable_price:.2f}")

# Compare to straight bond
straight_value = bond.value(hw_model, 0.0, yield_curve.get_rate(0.0))
print(f"Straight bond price: ${straight_value:.2f}")
print(f"Difference: ${straight_value - callable_price:.2f}")

# Show distribution of initial issuer values
print(f"\nDistribution of issuer values at t=0:")
print(f"Min: ${issuer_values[:, 0].min():.2f}")
print(f"25th percentile: ${np.percentile(issuer_values[:, 0], 25):.2f}")
print(f"Median: ${np.median(issuer_values[:, 0]):.2f}")
print(f"75th percentile: ${np.percentile(issuer_values[:, 0], 75):.2f}")
print(f"Max: ${issuer_values[:, 0].max():.2f}")