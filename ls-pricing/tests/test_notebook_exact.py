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
from ls_pricing.utils.curve_io import load_yield_curve_from_csv


def test_notebook_exact_reproduction():
    """Exactly reproduce the notebook setup"""

    # Load actual Treasury curve data (same as notebook)
    curve_date = datetime(2025, 4, 20)
    yield_curve = load_yield_curve_from_csv(
        "data/active_treasury_curve.csv",
        curve_date,
        tenor_col="Tenor",
        yield_col="Yield",
        encoding="latin-1",
    )

    # Same model parameters as notebook
    a = 0.03
    sigma = 0.008
    hw_model = HullWhiteModel(a=a, sigma=sigma, yield_curve=yield_curve)

    # Same MC parameters as notebook (5000 paths)
    mc_engine = MonteCarloEngine(model=hw_model, n_paths=5000, n_steps=60)

    # Create engine
    cb_engine = CallableBondEngine(
        mc_engine=mc_engine, basis_type="laguerre", basis_degree=3
    )

    # Same bond as notebook
    bond = CallableBond(
        face_value=100.0,
        coupon_rate=0.067,
        maturity=30.0,
        payment_frequency=2,
        first_call_date=10.0,
        call_price=100.0,
        credit_spread=0.01,
    )

    # Generate paths with same seed
    cb_engine.generate_paths(30.0, seed=42)

    # Price with from_investor_perspective=True (like notebook)
    result_investor = cb_engine.price_callable_bond(
        bond, from_investor_perspective=True
    )

    # Also price without specifying (like the test)
    result_default = cb_engine.price_callable_bond(bond)

    print("\n" + "=" * 60)
    print("Results with from_investor_perspective=True (notebook style):")
    print(f"  Callable: ${result_investor['callable_bond_price']:.2f}")
    print(f"  Straight: ${result_investor['straight_bond_price']:.2f}")
    print(f"  Option: ${result_investor['option_value']:.2f}")

    print("\nResults with default perspective (test style):")
    print(f"  Callable: ${result_default['callable_bond_price']:.2f}")
    print(f"  Straight: ${result_default['straight_bond_price']:.2f}")
    print(f"  Option: ${result_default['option_value']:.2f}")

    # Check if perspective matters
    if result_investor["callable_bond_price"] != result_default["callable_bond_price"]:
        print("\nWARNING: Perspective parameter changes the callable price!")

    # The key assertion
    assert (
        result_investor["callable_bond_price"] <= result_investor["straight_bond_price"]
    ), f"ERROR: Callable ({result_investor['callable_bond_price']:.2f}) > Straight ({result_investor['straight_bond_price']:.2f})"

    return result_investor


if __name__ == "__main__":
    test_notebook_exact_reproduction()
