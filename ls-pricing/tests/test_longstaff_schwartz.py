import pytest
import numpy as np
from datetime import datetime
from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.hull_white import HullWhiteModel
from ls_pricing.engine.monte_carlo import MonteCarloEngine
from ls_pricing.engine.longstaff_schwartz import LongstaffSchwartzEngine, RegressionBasis


class TestRegressionBasis:
    def test_laguerre_polynomials(self):
        x = np.array([0.0, 0.5, 1.0, 2.0])
        basis = RegressionBasis.laguerre_polynomials(x, degree=3)
        
        # Check shape
        assert basis.shape == (4, 4)
        
        # Check L0 = 1
        assert np.allclose(basis[:, 0], 1.0)
        
        # Check L1 = 1 - x
        expected_L1 = 1 - x
        assert np.allclose(basis[:, 1], expected_L1)
        
        # Check L2 = 1 - 2x + x²/2
        expected_L2 = 1 - 2*x + x**2/2
        assert np.allclose(basis[:, 2], expected_L2)
        
        # Check L3 = 1 - 3x + 3x²/2 - x³/6
        expected_L3 = 1 - 3*x + 3*x**2/2 - x**3/6
        assert np.allclose(basis[:, 3], expected_L3)
    
    def test_polynomial_basis(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        basis = RegressionBasis.polynomial_basis(x, degree=3)
        
        # Check shape
        assert basis.shape == (4, 4)
        
        # Check powers
        assert np.allclose(basis[:, 0], 1.0)  # x^0
        assert np.allclose(basis[:, 1], x)     # x^1
        assert np.allclose(basis[:, 2], x**2)  # x^2
        assert np.allclose(basis[:, 3], x**3)  # x^3


class TestLongstaffSchwartzEngine:
    def setup_method(self):
        # Create a simple yield curve
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        self.rates = np.array([0.03, 0.032, 0.035, 0.038, 0.04, 0.042])
        self.curve_date = datetime(2024, 1, 1)
        self.yield_curve = YieldCurve(self.tenors, self.rates, self.curve_date)
        
        # Create Hull-White model
        self.hw_model = HullWhiteModel(a=0.05, sigma=0.01, yield_curve=self.yield_curve)
        
        # Create Monte Carlo engine
        self.mc_engine = MonteCarloEngine(
            model=self.hw_model,
            n_paths=2000,
            n_steps=50
        )
        
        # Create LS engine
        self.ls_engine = LongstaffSchwartzEngine(
            mc_engine=self.mc_engine,
            basis_type='laguerre',
            basis_degree=3,
            regression_type='ols'
        )
    
    def test_init(self):
        assert self.ls_engine.mc_engine == self.mc_engine
        assert self.ls_engine.basis_type == 'laguerre'
        assert self.ls_engine.basis_degree == 3
        assert self.ls_engine.basis_func == RegressionBasis.laguerre_polynomials
    
    def test_payoff_calculation(self):
        bond_prices = np.array([0.95, 0.98, 1.02, 1.05])
        strike = 1.0
        
        # Test put payoff
        put_payoffs = self.ls_engine._calculate_payoff(bond_prices, strike, 'put')
        expected_put = np.array([0.05, 0.02, 0.0, 0.0])
        assert np.allclose(put_payoffs, expected_put)
        
        # Test call payoff
        call_payoffs = self.ls_engine._calculate_payoff(bond_prices, strike, 'call')
        expected_call = np.array([0.0, 0.0, 0.02, 0.05])
        assert np.allclose(call_payoffs, expected_call)
    
    def test_european_option_pricing(self):
        # Price a European put option
        result = self.ls_engine.price_european_option(
            strike=0.95,
            option_maturity=1.0,
            bond_maturity=2.0,
            option_type='put'
        )
        
        assert 'price' in result
        assert 'std_error' in result
        assert 'paths_used' in result
        
        # Price should be positive
        assert result['price'] > 0
        
        # Standard error should be small
        assert result['std_error'] < result['price'] * 0.1
        
        # Should use all paths
        assert result['paths_used'] == self.mc_engine.n_paths
    
    def test_american_option_pricing(self):
        # Price an American put option
        result = self.ls_engine.price_american_option(
            strike=0.95,
            option_maturity=1.0,
            bond_maturity=2.0,
            option_type='put'
        )
        
        assert 'price' in result
        assert 'std_error' in result
        assert 'exercise_boundary' in result
        assert 'exercise_prob' in result
        assert 'mean_exercise_time' in result
        assert 'paths_used' in result
        
        # Price should be positive
        assert result['price'] > 0
        
        # Exercise probability should be between 0 and 1
        assert 0 <= result['exercise_prob'] <= 1
        
        # Mean exercise time should be <= maturity
        if result['exercise_prob'] > 0:
            assert 0 < result['mean_exercise_time'] <= 1.0
    
    def test_american_vs_european(self):
        # Common parameters
        strike = 0.95
        option_maturity = 1.0
        bond_maturity = 2.0
        
        # Price both options
        american = self.ls_engine.price_american_option(
            strike=strike,
            option_maturity=option_maturity,
            bond_maturity=bond_maturity,
            option_type='put'
        )
        
        european = self.ls_engine.price_european_option(
            strike=strike,
            option_maturity=option_maturity,
            bond_maturity=bond_maturity,
            option_type='put'
        )
        
        # American should be worth at least as much as European
        assert american['price'] >= european['price'] - 1e-6
        
        # The difference should be reasonable (early exercise premium)
        premium = american['price'] - european['price']
        assert premium >= 0
        # For interest rate options, early exercise premium can be substantial
        # Just check it's not unreasonably large (e.g., more than the American price)
        assert premium < american['price']
    
    def test_put_call_relationship(self):
        # Test that puts and calls have reasonable relationship
        strike = 0.98
        option_maturity = 1.0
        bond_maturity = 2.0
        
        put_result = self.ls_engine.price_american_option(
            strike=strike,
            option_maturity=option_maturity,
            bond_maturity=bond_maturity,
            option_type='put'
        )
        
        call_result = self.ls_engine.price_american_option(
            strike=strike,
            option_maturity=option_maturity,
            bond_maturity=bond_maturity,
            option_type='call'
        )
        
        # Both should have positive prices
        assert put_result['price'] > 0
        assert call_result['price'] > 0
        
        # For American options, put-call parity doesn't hold exactly,
        # but prices should be in reasonable ranges
        assert put_result['price'] < strike  # Put can't be worth more than strike
        assert call_result['price'] < 1.0     # Call can't be worth more than bond
    
    def test_convergence_with_paths(self):
        # Test that price converges with more paths
        strike = 0.95
        option_maturity = 1.0
        bond_maturity = 2.0
        
        # Price with fewer paths
        mc_small = MonteCarloEngine(self.hw_model, n_paths=500, n_steps=30)
        ls_small = LongstaffSchwartzEngine(mc_small)
        result_small = ls_small.price_american_option(
            strike=strike,
            option_maturity=option_maturity,
            bond_maturity=bond_maturity,
            option_type='put'
        )
        
        # Price with more paths
        mc_large = MonteCarloEngine(self.hw_model, n_paths=5000, n_steps=30)
        ls_large = LongstaffSchwartzEngine(mc_large)
        result_large = ls_large.price_american_option(
            strike=strike,
            option_maturity=option_maturity,
            bond_maturity=bond_maturity,
            option_type='put'
        )
        
        # Standard error should decrease with more paths
        assert result_large['std_error'] < result_small['std_error']
        
        # Prices should be reasonably close
        price_diff = abs(result_large['price'] - result_small['price'])
        assert price_diff < 3 * max(result_small['std_error'], result_large['std_error'])
    
    def test_exercise_boundary(self):
        # Test that exercise boundary makes sense
        result = self.ls_engine.price_american_option(
            strike=0.95,
            option_maturity=2.0,
            bond_maturity=3.0,
            option_type='put'
        )
        
        exercise_boundary = result['exercise_boundary']
        
        if len(exercise_boundary) > 0:
            # Exercise should happen at reasonable bond prices
            for time, info in exercise_boundary.items():
                assert 0 < info['mean_bond_price'] < 1.2
                assert info['exercise_prob'] > 0
                assert info['n_exercised'] > 0
    
    def test_invalid_maturities(self):
        # Test error handling for invalid maturities
        with pytest.raises(ValueError, match="Bond maturity must be greater"):
            self.ls_engine.price_american_option(
                strike=0.95,
                option_maturity=2.0,
                bond_maturity=1.0,  # Invalid: bond matures before option
                option_type='put'
            )
    
    def test_different_basis_functions(self):
        # Test with polynomial basis
        ls_poly = LongstaffSchwartzEngine(
            mc_engine=self.mc_engine,
            basis_type='polynomial',
            basis_degree=3,
            regression_type='ols'
        )
        
        result_poly = ls_poly.price_american_option(
            strike=0.95,
            option_maturity=1.0,
            bond_maturity=2.0,
            option_type='put'
        )
        
        # Compare with Laguerre basis
        result_laguerre = self.ls_engine.price_american_option(
            strike=0.95,
            option_maturity=1.0,
            bond_maturity=2.0,
            option_type='put'
        )
        
        # Prices should be similar but not necessarily identical
        price_diff = abs(result_poly['price'] - result_laguerre['price'])
        assert price_diff < 0.01  # Within 1 cent
    
    def test_ridge_regression(self):
        # Test with Ridge regression
        ls_ridge = LongstaffSchwartzEngine(
            mc_engine=self.mc_engine,
            basis_type='laguerre',
            basis_degree=3,
            regression_type='ridge'
        )
        
        result = ls_ridge.price_american_option(
            strike=0.95,
            option_maturity=1.0,
            bond_maturity=2.0,
            option_type='put'
        )
        
        # Should still get reasonable price
        assert result['price'] > 0
        assert result['std_error'] < result['price'] * 0.1