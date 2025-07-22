#!/usr/bin/env python3
"""Debug callable bond pricing with focus on straight vs callable comparison"""

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

# Load CSV yield curve (reproducing notebook issue)
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

tenors = valid_data['TenorYears'].values
rates = valid_data['YieldDecimal'].values
curve_date = datetime(2025, 4, 20)
yield_curve = YieldCurve(tenors, rates, curve_date)

# Setup model
a = 0.03
sigma = 0.008
hw_model = HullWhiteModel(a=a, sigma=sigma, yield_curve=yield_curve)

# Create engines with different path counts
for n_paths in [100, 1000, 5000]:
    print(f"\n{'='*60}")
    print(f"Testing with {n_paths} paths")
    print('='*60)
    
    mc_engine = MonteCarloEngine(model=hw_model, n_paths=n_paths, n_steps=60)
    cb_engine = CallableBondEngine(mc_engine=mc_engine, basis_type='laguerre', basis_degree=3)
    
    # Create bond
    bond = CallableBond(
        face_value=100.0,
        coupon_rate=0.067,
        maturity=30.0,
        payment_frequency=2,
        first_call_date=10.0,
        call_price=100.0,
        credit_spread=0.01
    )
    
    # Price with fixed seed
    cb_engine.generate_paths(30.0, seed=42)
    result = cb_engine.price_callable_bond(bond, from_investor_perspective=True)
    
    print(f"\nResults:")
    print(f"  Callable: ${result['callable_bond_price']:.2f}")
    print(f"  Straight: ${result['straight_bond_price']:.2f}")
    print(f"  Option: ${result['option_value']:.2f}")
    print(f"  Call Prob: {result['exercise_probability']*100:.1f}%")
    
    # Check the relationship
    if result['callable_bond_price'] > result['straight_bond_price']:
        print("  ⚠️  WARNING: Callable > Straight (INCORRECT!)")
    else:
        print("  ✓ Callable < Straight (correct)")
    
    # Manual straight bond check
    r0 = yield_curve.get_rate(0.0)
    manual_straight = bond.value(hw_model, 0.0, r0)
    print(f"\nManual straight bond calc: ${manual_straight:.2f}")
    print(f"Engine straight bond calc: ${result['straight_bond_price']:.2f}")
    print(f"Difference: ${abs(manual_straight - result['straight_bond_price']):.2f}")

# Now test with hardcoded yield curve for comparison
print(f"\n\n{'='*60}")
print("Testing with HARDCODED yield curve (from passing test)")
print('='*60)

tenors_test = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
rates_test = np.array([0.043, 0.043, 0.041, 0.038, 0.038, 0.039, 0.041, 0.043, 0.046, 0.047])
yield_curve_test = YieldCurve(tenors_test, rates_test, curve_date)

hw_model_test = HullWhiteModel(a=a, sigma=sigma, yield_curve=yield_curve_test)
mc_engine_test = MonteCarloEngine(model=hw_model_test, n_paths=5000, n_steps=60)
cb_engine_test = CallableBondEngine(mc_engine_test, basis_type='laguerre', basis_degree=3)

cb_engine_test.generate_paths(30.0, seed=42)
result_test = cb_engine_test.price_callable_bond(bond, from_investor_perspective=True)

print(f"\nResults with hardcoded curve:")
print(f"  Callable: ${result_test['callable_bond_price']:.2f}")
print(f"  Straight: ${result_test['straight_bond_price']:.2f}")
print(f"  Option: ${result_test['option_value']:.2f}")
print(f"  Call Prob: {result_test['exercise_probability']*100:.1f}%")

if result_test['callable_bond_price'] > result_test['straight_bond_price']:
    print("  ⚠️  WARNING: Callable > Straight (INCORRECT!)")
else:
    print("  ✓ Callable < Straight (correct)")