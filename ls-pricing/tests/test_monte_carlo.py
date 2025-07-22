import pytest
import numpy as np
from datetime import datetime
from ls_pricing.core.curves import YieldCurve
from ls_pricing.core.hull_white import HullWhiteModel
from ls_pricing.engine.monte_carlo import MonteCarloEngine


class TestMonteCarloEngine:
    def setup_method(self):
        # Create a simple yield curve
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        self.rates = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045])
        self.curve_date = datetime(2024, 1, 1)
        self.yield_curve = YieldCurve(self.tenors, self.rates, self.curve_date)
        
        # Create Hull-White model
        self.hw_model = HullWhiteModel(a=0.05, sigma=0.01, yield_curve=self.yield_curve)
        
        # Create Monte Carlo engine
        self.mc_engine = MonteCarloEngine(
            model=self.hw_model,
            n_paths=1000,
            n_steps=50
        )
    
    def test_init(self):
        assert self.mc_engine.model == self.hw_model
        assert self.mc_engine.n_paths == 1000
        assert self.mc_engine.n_steps == 50
        assert self.mc_engine.paths_cache is None
    
    def test_generate_paths_shape(self):
        T = 2.0
        paths = self.mc_engine.generate_paths(T=T, seed=42)
        
        assert 'rates' in paths
        assert 'times' in paths
        assert 'discount_factors' in paths
        
        # Check shapes
        assert paths['rates'].shape == (1000, 51)  # n_paths x (n_steps + 1)
        assert paths['times'].shape == (51,)
        assert paths['discount_factors'].shape == (1000, 51)
    
    def test_generate_paths_time_grid(self):
        T = 2.0
        paths = self.mc_engine.generate_paths(T=T, seed=42)
        
        # Check time grid
        assert paths['times'][0] == 0.0
        assert paths['times'][-1] == T
        assert np.allclose(np.diff(paths['times']), T / self.mc_engine.n_steps)
    
    def test_generate_paths_initial_values(self):
        T = 2.0
        paths = self.mc_engine.generate_paths(T=T, seed=42)
        
        # Check initial rate
        r0 = self.yield_curve.get_rate(0.0)
        assert np.allclose(paths['rates'][:, 0], r0)
        
        # Check initial discount factor
        assert np.allclose(paths['discount_factors'][:, 0], 1.0)
    
    def test_generate_paths_discount_factors(self):
        T = 2.0
        paths = self.mc_engine.generate_paths(T=T, seed=42)
        
        # Discount factors should be between 0 and 1
        assert np.all(paths['discount_factors'] > 0)
        assert np.all(paths['discount_factors'] <= 1)
        
        # Discount factors should be decreasing along each path
        for i in range(10):  # Check first 10 paths
            assert np.all(np.diff(paths['discount_factors'][i]) <= 0)
    
    def test_caching_functionality(self):
        T = 2.0
        
        # Generate paths without cache
        paths1 = self.mc_engine.generate_paths(T=T, seed=42, use_cache=False)
        assert self.mc_engine.paths_cache is None
        
        # Generate paths with cache
        paths2 = self.mc_engine.generate_paths(T=T, seed=42, use_cache=True)
        assert self.mc_engine.paths_cache is not None
        
        # Retrieve cached paths
        paths3 = self.mc_engine.generate_paths(T=T, seed=123, use_cache=True)
        
        # paths3 should be identical to paths2 (from cache)
        assert np.array_equal(paths3['rates'], paths2['rates'])
        assert np.array_equal(paths3['times'], paths2['times'])
        assert np.array_equal(paths3['discount_factors'], paths2['discount_factors'])
    
    def test_path_statistics(self):
        T = 2.0
        paths = self.mc_engine.generate_paths(T=T, seed=42)
        
        stats = self.mc_engine.get_path_statistics(paths)
        
        assert 'mean_terminal_rate' in stats
        assert 'std_terminal_rate' in stats
        assert 'mean_path' in stats
        assert 'percentiles' in stats
        
        # Check statistics make sense
        assert isinstance(stats['mean_terminal_rate'], (float, np.floating))
        assert stats['std_terminal_rate'] > 0
        assert len(stats['mean_path']) == 51  # n_steps + 1
        assert len(stats['percentiles']) == 3  # 5th, 50th, 95th
        
        # Percentiles should be ordered
        assert stats['percentiles'][0] < stats['percentiles'][1] < stats['percentiles'][2]
    
    def test_reproducibility(self):
        T = 2.0
        
        # Generate paths with same seed
        paths1 = self.mc_engine.generate_paths(T=T, seed=42)
        paths2 = self.mc_engine.generate_paths(T=T, seed=42)
        
        # Should be identical
        assert np.array_equal(paths1['rates'], paths2['rates'])
        assert np.array_equal(paths1['discount_factors'], paths2['discount_factors'])
    
    def test_different_seeds(self):
        T = 2.0
        
        # Generate paths with different seeds
        paths1 = self.mc_engine.generate_paths(T=T, seed=42)
        paths2 = self.mc_engine.generate_paths(T=T, seed=123)
        
        # Should be different
        assert not np.array_equal(paths1['rates'], paths2['rates'])
        
        # But statistics should be similar
        stats1 = self.mc_engine.get_path_statistics(paths1)
        stats2 = self.mc_engine.get_path_statistics(paths2)
        
        assert abs(stats1['mean_terminal_rate'] - stats2['mean_terminal_rate']) < 0.01
    
    def test_large_simulation(self):
        # Test with more paths
        large_mc = MonteCarloEngine(
            model=self.hw_model,
            n_paths=10000,
            n_steps=100
        )
        
        T = 5.0
        paths = large_mc.generate_paths(T=T, seed=42)
        
        assert paths['rates'].shape == (10000, 101)
        
        # With more paths, statistics should be more stable
        stats = large_mc.get_path_statistics(paths)
        
        # Mean should converge to long-term level
        long_term_rate = self.yield_curve.get_rate(T)
        assert abs(stats['mean_terminal_rate'] - long_term_rate) < 0.05