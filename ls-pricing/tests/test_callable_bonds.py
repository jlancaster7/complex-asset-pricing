import pytest
import numpy as np
from datetime import datetime
from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.hull_white import HullWhiteModel
from ls_pricing.engine.monte_carlo import MonteCarloEngine
from ls_pricing.engine.callable_bond import CallableBondEngine
from ls_pricing.instruments.bonds import CouponBond, CallableBond


class TestCouponBond:
    def setup_method(self):
        # Create a simple yield curve
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
        self.rates = np.array([0.03, 0.032, 0.035, 0.038, 0.04, 0.042, 0.045, 0.047])
        self.curve_date = datetime(2024, 1, 1)
        self.yield_curve = YieldCurve(self.tenors, self.rates, self.curve_date)
        
        # Create Hull-White model
        self.hw_model = HullWhiteModel(a=0.05, sigma=0.01, yield_curve=self.yield_curve)
        
        # Create a test bond
        self.bond = CouponBond(
            face_value=100.0,
            coupon_rate=0.05,
            maturity=10.0,
            payment_frequency=2
        )
    
    def test_cash_flow_schedule(self):
        times, flows = self.bond.get_cash_flow_schedule()
        
        # Should have 20 payments (10 years * 2 payments per year)
        assert len(times) == 20
        assert len(flows) == 20
        
        # First 19 should be coupon only
        expected_coupon = 100.0 * 0.05 / 2  # 2.5
        assert np.allclose(flows[:-1], expected_coupon)
        
        # Last should be coupon + principal
        assert flows[-1] == expected_coupon + 100.0
        
        # Times should be evenly spaced
        assert np.allclose(np.diff(times), 0.5)
    
    def test_cash_flow_schedule_from_time(self):
        # Get cash flows starting from year 5
        times, flows = self.bond.get_cash_flow_schedule(from_time=5.0)
        
        # Should have 10 payments left
        assert len(times) == 10
        assert times[0] == 5.5  # Next payment after year 5
    
    def test_bond_value(self):
        # Value at t=0
        value = self.bond.value(self.hw_model, 0.0, 0.035)
        
        # With 5% coupon and ~3.5% yield curve, should trade above par
        assert value > 100.0
        assert value < 120.0  # Reasonable upper bound
    
    def test_bond_value_at_maturity(self):
        # Value just before maturity
        value = self.bond.value(self.hw_model, 9.99, 0.04)
        
        # Should be close to final payment
        final_payment = 100.0 + 2.5  # Principal + last coupon
        assert abs(value - final_payment) < 1.0
    
    def test_yield_to_maturity(self):
        # Price the bond
        price = 105.0
        
        # Calculate YTM
        ytm = self.bond.yield_to_maturity(price)
        
        # YTM should be less than coupon rate since trading above par
        assert ytm < 0.05
        assert ytm > 0.0
        
        # Verify by repricing
        times, flows = self.bond.get_cash_flow_schedule()
        recalc_price = sum(cf * np.exp(-ytm * t) for t, cf in zip(times, flows))
        assert abs(recalc_price - price) < 0.01
    
    def test_credit_spread_effect(self):
        # Create two bonds: one with no spread, one with 100bp spread
        bond_no_spread = CouponBond(
            face_value=100.0,
            coupon_rate=0.05,
            maturity=10.0,
            payment_frequency=2,
            credit_spread=0.0
        )
        
        bond_with_spread = CouponBond(
            face_value=100.0,
            coupon_rate=0.05,
            maturity=10.0,
            payment_frequency=2,
            credit_spread=0.01  # 100bp
        )
        
        # Value both bonds
        value_no_spread = bond_no_spread.value(self.hw_model, 0.0, 0.035)
        value_with_spread = bond_with_spread.value(self.hw_model, 0.0, 0.035)
        
        # Bond with spread should be worth less
        assert value_with_spread < value_no_spread
        
        # The difference should be significant for a 10-year bond
        assert (value_no_spread - value_with_spread) > 5.0  # More than $5 difference


class TestCallableBond:
    def setup_method(self):
        self.callable_bond = CallableBond(
            face_value=100.0,
            coupon_rate=0.05,
            maturity=30.0,
            payment_frequency=2,
            first_call_date=10.0,
            call_price=100.0
        )
    
    def test_is_callable(self):
        assert not self.callable_bond.is_callable(5.0)
        assert not self.callable_bond.is_callable(9.99)
        assert self.callable_bond.is_callable(10.0)
        assert self.callable_bond.is_callable(20.0)
    
    def test_get_exercise_times(self):
        time_grid = np.linspace(0, 30, 61)  # Every 0.5 years
        
        exercise_indices = self.callable_bond.get_exercise_times(time_grid)
        
        # Should only include times >= 10 and < 30
        exercise_times = [time_grid[i] for i in exercise_indices]
        assert min(exercise_times) >= 10.0
        assert max(exercise_times) < 30.0
        
        # Should have 40 exercise opportunities (years 10-29.5)
        assert len(exercise_indices) == 40


class TestCallableBondEngine:
    def setup_method(self):
        # Create yield curve
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
        self.rates = np.array([0.03, 0.032, 0.035, 0.038, 0.04, 0.042, 0.045, 0.047])
        self.curve_date = datetime(2024, 1, 1)
        self.yield_curve = YieldCurve(self.tenors, self.rates, self.curve_date)
        
        # Create Hull-White model
        self.hw_model = HullWhiteModel(a=0.05, sigma=0.01, yield_curve=self.yield_curve)
        
        # Create Monte Carlo engine
        self.mc_engine = MonteCarloEngine(
            model=self.hw_model,
            n_paths=2000,
            n_steps=60  # Semi-annual steps for 30 years
        )
        
        # Create callable bond engine
        self.cb_engine = CallableBondEngine(self.mc_engine)
        
        # Create test callable bond
        self.callable_bond = CallableBond(
            face_value=100.0,
            coupon_rate=0.05,
            maturity=30.0,
            payment_frequency=2,
            first_call_date=10.0,
            call_price=100.0
        )
    
    def test_price_callable_bond_structure(self):
        result = self.cb_engine.price_callable_bond(self.callable_bond)
        
        # Check all expected fields are present
        assert 'callable_bond_price' in result
        assert 'option_value' in result
        assert 'straight_bond_price' in result
        assert 'exercise_probability' in result
        assert 'mean_call_time' in result
        assert 'exercise_boundary' in result
        assert 'paths_used' in result
    
    def test_callable_less_than_straight(self):
        result = self.cb_engine.price_callable_bond(self.callable_bond)
        
        # Callable bond should be worth less than straight bond (from investor perspective)
        # Allow small numerical tolerance due to Monte Carlo noise
        assert result['callable_bond_price'] <= result['straight_bond_price'] + 0.5
        
        # Option value should be positive
        assert result['option_value'] > 0
        
        # Option value should equal the difference
        assert np.isclose(
            result['option_value'],
            result['straight_bond_price'] - result['callable_bond_price'],
            rtol=1e-6
        )
    
    def test_exercise_probability_reasonable(self):
        result = self.cb_engine.price_callable_bond(self.callable_bond)
        
        # Exercise probability should be between 0 and 1
        assert 0 <= result['exercise_probability'] <= 1
        
        # If exercised, mean call time should be after first call date
        if result['exercise_probability'] > 0:
            assert result['mean_call_time'] >= self.callable_bond.first_call_date
            assert result['mean_call_time'] <= self.callable_bond.maturity
    
    def test_high_coupon_more_likely_called(self):
        # High coupon bond (more likely to be called)
        high_coupon_bond = CallableBond(
            face_value=100.0,
            coupon_rate=0.07,  # 7% coupon when rates are ~4%
            maturity=30.0,
            payment_frequency=2,
            first_call_date=10.0,
            call_price=100.0
        )
        
        # Low coupon bond (less likely to be called)
        low_coupon_bond = CallableBond(
            face_value=100.0,
            coupon_rate=0.03,  # 3% coupon
            maturity=30.0,
            payment_frequency=2,
            first_call_date=10.0,
            call_price=100.0
        )
        
        high_result = self.cb_engine.price_callable_bond(high_coupon_bond)
        low_result = self.cb_engine.price_callable_bond(low_coupon_bond)
        
        # High coupon bond should have higher call probability
        assert high_result['exercise_probability'] > low_result['exercise_probability']
        
        # High coupon bond should have larger option value
        assert high_result['option_value'] > low_result['option_value']
    
    def test_issuer_perspective(self):
        # From investor perspective
        investor_result = self.cb_engine.price_callable_bond(
            self.callable_bond,
            from_investor_perspective=True
        )
        
        # From issuer perspective
        issuer_result = self.cb_engine.price_callable_bond(
            self.callable_bond,
            from_investor_perspective=False
        )
        
        # Prices and probabilities should be the same
        assert investor_result['callable_bond_price'] == issuer_result['callable_bond_price']
        assert investor_result['exercise_probability'] == issuer_result['exercise_probability']
        
        # Option values should have opposite signs
        assert np.isclose(
            investor_result['option_value'],
            -issuer_result['option_value'],
            rtol=1e-6
        )
    
    def test_spread_affects_call_probability(self):
        # Create two callable bonds with same coupon but different spreads
        bond_low_spread = CallableBond(
            face_value=100.0,
            coupon_rate=0.05,
            maturity=30.0,
            payment_frequency=2,
            first_call_date=10.0,
            call_price=100.0,
            credit_spread=0.005  # 50bp spread
        )
        
        bond_high_spread = CallableBond(
            face_value=100.0,
            coupon_rate=0.05,
            maturity=30.0,
            payment_frequency=2,
            first_call_date=10.0,
            call_price=100.0,
            credit_spread=0.015  # 150bp spread
        )
        
        # Price both bonds
        result_low = self.cb_engine.price_callable_bond(bond_low_spread)
        result_high = self.cb_engine.price_callable_bond(bond_high_spread)
        
        # Higher spread bond should have lower call probability
        # (because refinancing at risk-free + spread is more expensive)
        assert result_high['exercise_probability'] < result_low['exercise_probability']
        
        # Higher spread bond should have lower option value
        assert result_high['option_value'] < result_low['option_value']
        
        # Higher spread bond should trade at lower price
        assert result_high['callable_bond_price'] < result_low['callable_bond_price']
    
    def test_no_call_before_lockout(self):
        # Bond with very short lockout
        short_lockout_bond = CallableBond(
            face_value=100.0,
            coupon_rate=0.06,
            maturity=10.0,
            payment_frequency=2,
            first_call_date=10.0,  # Callable only at maturity (effectively non-callable)
            call_price=100.0
        )
        
        mc_engine_short = MonteCarloEngine(
            model=self.hw_model,
            n_paths=1000,
            n_steps=20
        )
        cb_engine_short = CallableBondEngine(mc_engine_short)
        
        result = cb_engine_short.price_callable_bond(short_lockout_bond)
        
        # Should have very low call probability with such short window
        assert result['exercise_probability'] < 0.5
        
        # If called, must be after first call date
        if result['mean_call_time'] > 0:
            assert result['mean_call_time'] >= short_lockout_bond.first_call_date