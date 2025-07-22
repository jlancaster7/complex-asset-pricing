import numpy as np
from typing import Dict, Optional
from ..core.hull_white import HullWhiteModel


class MonteCarloEngine:
    """Wrapper for Monte Carlo simulation with diagnostics"""
    
    def __init__(
        self, 
        model: HullWhiteModel, 
        n_paths: int = 10000,
        n_steps: int = 100
    ):
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.paths_cache = None
        
    def generate_paths(
        self, 
        T: float, 
        seed: Optional[int] = None,
        use_cache: bool = False
    ) -> Dict[str, np.ndarray]:
        """Generate paths with optional caching"""
        
        if use_cache and self.paths_cache is not None:
            return self.paths_cache
            
        # Generate time grid
        t_grid = np.linspace(0, T, self.n_steps + 1)
        
        # Simulate short rate paths
        rates = self.model.simulate_short_rate(
            t_grid=t_grid,
            n_paths=self.n_paths,
            seed=seed
        )
        
        # Calculate discount factors along paths
        # Simple approximation: D(0,t) ≈ exp(-∫r(s)ds)
        dt = T / self.n_steps
        cumulative_rates = np.cumsum(rates[:, :-1] * dt, axis=1)
        discount_factors = np.exp(-cumulative_rates)
        discount_factors = np.column_stack([np.ones(self.n_paths), discount_factors])
        
        paths = {
            'rates': rates,
            'times': t_grid,
            'discount_factors': discount_factors
        }
        
        if use_cache:
            self.paths_cache = paths
            
        return paths
    
    def get_path_statistics(self, paths: Dict[str, np.ndarray]) -> Dict:
        """Calculate basic statistics for validation"""
        rates = paths['rates']
        
        return {
            'mean_terminal_rate': np.mean(rates[:, -1]),
            'std_terminal_rate': np.std(rates[:, -1]),
            'mean_path': np.mean(rates, axis=0),
            'percentiles': np.percentile(rates[:, -1], [5, 50, 95])
        }