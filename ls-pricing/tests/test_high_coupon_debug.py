#!/usr/bin/env python3
"""Debug test for high coupon callable bond pricing issue"""

import pytest
import numpy as np
from datetime import datetime
import sys
import pandas as pd

from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.hull_white import HullWhiteModel
from ls_pricing.engine.monte_carlo import MonteCarloEngine
from ls_pricing.engine.callable_bond import CallableBondEngine
from ls_pricing.instruments.bonds import CallableBond


def get_yield_curve():
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
    return yield_curve


def test_high_coupon_callable_pricing():
    """Test that callable bonds with high coupons price correctly"""

    yield_curve = get_yield_curve()

    # tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    # rates = np.array(
    #     [0.04305, 0.04244, 0.04052, 0.03827, 0.03879, 0.04337, 0.04893, 0.04901]
    # )
    # curve_date = datetime(2025, 7, 22)
    # yield_curve = YieldCurve(tenors, rates, curve_date)
    # print(yield_curve.get_rate(10))

    # Model parameters
    a = 0.03  # Lower mean reversion for munis
    sigma = 0.008  # Lower volatility for munis

    # Create Hull-White model
    hw_model = HullWhiteModel(a=a, sigma=sigma, yield_curve=yield_curve)

    # MC parameters
    n_paths = 5000
    n_steps = 60  # Semi-annual steps for 30 years

    # Create MC engine
    mc_engine = MonteCarloEngine(model=hw_model, n_paths=n_paths, n_steps=n_steps)

    # Bond parameters - matching notebook
    face_value = 100
    coupon_rate = 0.077  # 4.7%
    maturity = 30
    payment_frequency = 2
    credit_spread = 0.01  # 100 bps
    call_price = 100  # Par call
    first_call_date = 10  # Callable after 10 years

    # Create callable bond
    bond = CallableBond(
        face_value=face_value,
        coupon_rate=coupon_rate,
        maturity=maturity,
        payment_frequency=payment_frequency,
        credit_spread=credit_spread,
        call_price=call_price,
        first_call_date=first_call_date,
    )

    # Create engine and price
    cb_engine = CallableBondEngine(
        mc_engine=mc_engine, basis_type="laguerre", basis_degree=3
    )

    # Pre-generate paths for consistency
    print("\nGenerating paths...")
    cb_engine.generate_paths(maturity, seed=42)

    # Price the callable bond
    print("\nPricing callable bond...")
    results = cb_engine.price_callable_bond(bond)

    print("\n" + "=" * 50)
    print("High Coupon Callable Bond Test Results:")
    print("=" * 50)
    print(f"Coupon Rate: {coupon_rate*100:.1f}%")
    print(
        f"10Y Treasury + Spread: {(yield_curve.get_rate(10.0) + credit_spread)*100:.1f}%"
    )
    print(f"\nPricing Results:")
    print(f"  Callable Bond Price: ${results['callable_bond_price']:.2f}")
    print(f"  Straight Bond Price: ${results['straight_bond_price']:.2f}")
    print(f"  Option Value: ${results['option_value']:.2f}")
    print(f"  Call Probability: {results['exercise_probability']*100:.1f}%")

    # IMPORTANT TEST: Callable should be worth LESS than straight bond
    assert (
        results["callable_bond_price"] <= results["straight_bond_price"]
    ), f"ERROR: Callable price ({results['callable_bond_price']:.2f}) > Straight price ({results['straight_bond_price']:.2f})"

    # Option value should be positive
    assert (
        results["option_value"] >= 0
    ), f"ERROR: Option value is negative: {results['option_value']:.2f}"

    # For a high coupon bond, call probability should be high
    # assert (
    #     results["exercise_probability"] > 0.5
    # ), f"ERROR: Call probability too low for high coupon bond: {results['exercise_probability']*100:.1f}%"

    # Let's also trace through some intermediate calculations
    print("\n" + "=" * 50)
    print("Debugging Information:")
    print("=" * 50)

    # Check initial rate
    r0 = yield_curve.get_rate(0.0)
    print(f"Initial rate (r0): {r0*100:.2f}%")
    print(f"With credit spread: {(r0 + credit_spread)*100:.2f}%")

    # Manual straight bond calculation
    discount_rate = r0 + credit_spread
    coupon_payment = face_value * coupon_rate / payment_frequency
    n_payments = int(maturity * payment_frequency)

    pv = 0
    for i in range(1, n_payments + 1):
        t = i / payment_frequency
        pv += coupon_payment * np.exp(-discount_rate * t)
    pv += face_value * np.exp(-discount_rate * maturity)

    print(f"\nManual PV calculation: ${pv:.2f}")
    print(f"Engine straight bond price: ${results['straight_bond_price']:.2f}")

    # The callable should trade at a significant discount for high coupon bonds
    discount_pct = (
        (results["straight_bond_price"] - results["callable_bond_price"])
        / results["straight_bond_price"]
        * 100
    )
    print(f"\nCallable trades at {discount_pct:.1f}% discount to straight bond")

    return results


if __name__ == "__main__":
    test_high_coupon_callable_pricing()
