#!/usr/bin/env python3
"""Test that exactly reproduces notebook setup to identify pricing issue"""

import numpy as np
import pandas as pd
from datetime import datetime
from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.hull_white import HullWhiteModel
from ls_pricing.engine.monte_carlo import MonteCarloEngine
from ls_pricing.engine.callable_bond import CallableBondEngine
from ls_pricing.instruments.bonds import CallableBond


def test_notebook_exact_reproduction():
    """Exactly reproduce the notebook setup"""
    
    # Load actual Treasury curve data (same as notebook)
    treasury_data = pd.read_csv('data/active_treasury_curve.csv', encoding='latin-1')
    
    # Parse tenor strings to years (same function as notebook)
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
    
    # Process data exactly as notebook
    treasury_data['TenorYears'] = treasury_data['Tenor'].apply(tenor_to_years)
    treasury_data['YieldDecimal'] = treasury_data['Yield'] / 100.0
    valid_data = treasury_data.dropna(subset=['TenorYears']).sort_values('TenorYears')
    
    tenors = valid_data['TenorYears'].values
    rates = valid_data['YieldDecimal'].values
    
    print(f"\nLoaded {len(tenors)} tenor points from CSV")
    print(f"Tenors: {tenors}")
    print(f"Rates: {rates}")
    
    # Create yield curve (same date as notebook)
    curve_date = datetime(2025, 4, 20)
    yield_curve = YieldCurve(tenors, rates, curve_date)
    
    # Same model parameters as notebook
    a = 0.03
    sigma = 0.008
    hw_model = HullWhiteModel(a=a, sigma=sigma, yield_curve=yield_curve)
    
    # Same MC parameters as notebook (5000 paths)
    mc_engine = MonteCarloEngine(model=hw_model, n_paths=5000, n_steps=60)
    
    # Create engine
    cb_engine = CallableBondEngine(
        mc_engine=mc_engine,
        basis_type='laguerre',
        basis_degree=3
    )
    
    # Same bond as notebook
    bond = CallableBond(
        face_value=100.0,
        coupon_rate=0.067,
        maturity=30.0,
        payment_frequency=2,
        first_call_date=10.0,
        call_price=100.0,
        credit_spread=0.01
    )
    
    # Generate paths with same seed
    cb_engine.generate_paths(30.0, seed=42)
    
    # Price with from_investor_perspective=True (like notebook)
    result_investor = cb_engine.price_callable_bond(bond, from_investor_perspective=True)
    
    # Also price without specifying (like the test)
    result_default = cb_engine.price_callable_bond(bond)
    
    print("\n" + "="*60)
    print("Results with from_investor_perspective=True (notebook style):")
    print(f"  Callable: ${result_investor['callable_bond_price']:.2f}")
    print(f"  Straight: ${result_investor['straight_bond_price']:.2f}")
    print(f"  Option: ${result_investor['option_value']:.2f}")
    
    print("\nResults with default perspective (test style):")
    print(f"  Callable: ${result_default['callable_bond_price']:.2f}")
    print(f"  Straight: ${result_default['straight_bond_price']:.2f}")
    print(f"  Option: ${result_default['option_value']:.2f}")
    
    # Check if perspective matters
    if result_investor['callable_bond_price'] != result_default['callable_bond_price']:
        print("\nWARNING: Perspective parameter changes the callable price!")
    
    # The key assertion
    assert result_investor['callable_bond_price'] <= result_investor['straight_bond_price'], \
        f"ERROR: Callable ({result_investor['callable_bond_price']:.2f}) > Straight ({result_investor['straight_bond_price']:.2f})"
    
    return result_investor


if __name__ == "__main__":
    test_notebook_exact_reproduction()