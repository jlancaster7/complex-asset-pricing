import pytest
import numpy as np
from datetime import datetime
from ls_pricing.core.curves import YieldCurve


class TestYieldCurve:
    def setup_method(self):
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        self.rates = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045])
        self.curve_date = datetime(2024, 1, 1)
        self.yield_curve = YieldCurve(self.tenors, self.rates, self.curve_date)
    
    def test_init_valid(self):
        assert np.array_equal(self.yield_curve.tenors, self.tenors)
        assert np.array_equal(self.yield_curve.rates, self.rates)
        assert self.yield_curve.curve_date == self.curve_date
        assert self.yield_curve.day_count == "ACT/365"
    
    def test_init_invalid_length(self):
        with pytest.raises(ValueError, match="same length"):
            YieldCurve(self.tenors[:-1], self.rates, self.curve_date)
    
    def test_init_non_increasing_tenors(self):
        bad_tenors = np.array([0.25, 1.0, 0.5, 2.0])
        bad_rates = np.array([0.02, 0.03, 0.025, 0.035])
        with pytest.raises(ValueError, match="strictly increasing"):
            YieldCurve(bad_tenors, bad_rates, self.curve_date)
    
    def test_get_rate_interpolation(self):
        rate_0_75 = self.yield_curve.get_rate(0.75)
        assert 0.025 < rate_0_75 < 0.03
        
        rate_3_0 = self.yield_curve.get_rate(3.0)
        assert 0.035 < rate_3_0 < 0.04
    
    def test_get_rate_exact_tenor(self):
        for tenor, expected_rate in zip(self.tenors, self.rates):
            assert np.isclose(self.yield_curve.get_rate(tenor), expected_rate)
    
    def test_get_rate_array(self):
        test_tenors = np.array([0.5, 1.0, 2.0])
        rates = self.yield_curve.get_rate(test_tenors)
        assert len(rates) == len(test_tenors)
        assert np.allclose(rates, [0.025, 0.03, 0.035])
    
    def test_get_discount_factor(self):
        df_1y = self.yield_curve.get_discount_factor(1.0)
        expected_df = np.exp(-0.03 * 1.0)
        assert np.isclose(df_1y, expected_df)
        
        df_0 = self.yield_curve.get_discount_factor(0.0)
        assert np.isclose(df_0, 1.0)
    
    def test_get_forward_rate(self):
        fwd_rate = self.yield_curve.get_forward_rate(1.0, 2.0)
        
        df_1 = self.yield_curve.get_discount_factor(1.0)
        df_2 = self.yield_curve.get_discount_factor(2.0)
        expected_fwd = -np.log(df_2 / df_1) / (2.0 - 1.0)
        
        assert np.isclose(fwd_rate, expected_fwd)
    
    def test_get_forward_rate_invalid(self):
        with pytest.raises(ValueError, match="Start tenor must be less"):
            self.yield_curve.get_forward_rate(2.0, 1.0)
    
    def test_shift_curve(self):
        shift = 0.01
        shifted_curve = self.yield_curve.shift_curve(shift)
        
        assert np.allclose(shifted_curve.rates, self.rates + shift)
        assert np.array_equal(shifted_curve.tenors, self.tenors)
        assert shifted_curve.curve_date == self.curve_date