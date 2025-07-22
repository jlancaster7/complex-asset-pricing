import pytest
import numpy as np
from datetime import datetime
from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.hull_white import HullWhiteModel


class TestHullWhiteModel:
    def setup_method(self):
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        self.rates = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045])
        self.curve_date = datetime(2024, 1, 1)
        self.yield_curve = YieldCurve(self.tenors, self.rates, self.curve_date)
        
        self.a = 0.05
        self.sigma = 0.01
        self.hw_model = HullWhiteModel(self.a, self.sigma, self.yield_curve)
    
    def test_init_valid(self):
        assert self.hw_model.a == self.a
        assert self.hw_model.sigma == self.sigma
        assert self.hw_model.yield_curve == self.yield_curve
    
    def test_init_invalid_params(self):
        with pytest.raises(ValueError, match="positive"):
            HullWhiteModel(-0.05, 0.01, self.yield_curve)
        
        with pytest.raises(ValueError, match="positive"):
            HullWhiteModel(0.05, -0.01, self.yield_curve)
    
    def test_simulate_short_rate_shape(self):
        t_grid = np.linspace(0, 2, 21)
        n_paths = 1000
        
        r_paths = self.hw_model.simulate_short_rate(t_grid, n_paths, seed=42)
        
        assert r_paths.shape == (n_paths, len(t_grid))
        assert np.all(np.isfinite(r_paths))
    
    def test_simulate_short_rate_initial_value(self):
        t_grid = np.linspace(0, 2, 21)
        n_paths = 100
        
        r_paths = self.hw_model.simulate_short_rate(t_grid, n_paths, seed=42)
        
        r0 = self.yield_curve.get_rate(0.0)
        assert np.allclose(r_paths[:, 0], r0)
    
    def test_simulate_short_rate_mean_reversion(self):
        t_grid = np.linspace(0, 10, 101)
        n_paths = 10000
        
        r_paths = self.hw_model.simulate_short_rate(t_grid, n_paths, seed=42)
        
        mean_path = np.mean(r_paths, axis=0)
        long_term_mean = mean_path[-20:].mean()
        long_term_rate = self.yield_curve.get_rate(10.0)
        
        assert abs(long_term_mean - long_term_rate) < 0.03
    
    def test_zero_bond_price(self):
        t = 1.0
        T = 2.0
        r_t = 0.03
        
        bond_price = self.hw_model.zero_bond_price(t, T, r_t)
        
        assert 0 < bond_price < 1
        assert np.isfinite(bond_price)
    
    def test_zero_bond_price_immediate_maturity(self):
        t = 1.0
        T = 1.0
        r_t = 0.03
        
        bond_price = self.hw_model.zero_bond_price(t, T, r_t)
        assert np.isclose(bond_price, 1.0)
    
    def test_zero_bond_price_consistency(self):
        t = 0.0
        T = 1.0
        r_0 = self.yield_curve.get_rate(0.0)
        
        bond_price = self.hw_model.zero_bond_price(t, T, r_0)
        market_df = self.yield_curve.get_discount_factor(T)
        
        assert np.isclose(bond_price, market_df, rtol=1e-6)
    
    def test_calibrate_placeholder(self):
        swaption_data = {}
        a_cal, sigma_cal = self.hw_model.calibrate_to_swaptions(swaption_data)
        
        assert a_cal == self.a
        assert sigma_cal == self.sigma