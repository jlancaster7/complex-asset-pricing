import numpy as np
from typing import Dict, Optional, Protocol


class ShortRateModel(Protocol):
    def simulate_short_rate(
        self, t_grid: np.ndarray, n_paths: int, seed: Optional[int] = None
    ) -> np.ndarray: ...


class MonteCarloEngine:
    """Wrapper for Monte Carlo simulation with diagnostics"""

    def __init__(self, model: ShortRateModel, n_paths: int = 10000, n_steps: int = 100):
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.paths_cache: Optional[Dict[str, np.ndarray]] = None

    def generate_paths(
        self,
        T: float,
        seed: Optional[int] = None,
        use_cache: bool = False,
        time_grid: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate paths with optional caching and custom time grid.

        If time_grid is provided, it overrides the uniform grid implied by n_steps.
        Discount factors are computed using pathwise integrals with the actual step sizes.
        """

        if use_cache and self.paths_cache is not None:
            return self.paths_cache

        # Generate time grid
        if time_grid is None:
            t_grid = np.linspace(0, T, self.n_steps + 1)
        else:
            t_grid = np.asarray(time_grid, dtype=float)
            if t_grid[0] != 0.0:
                t_grid = np.concatenate([[0.0], t_grid])
            if t_grid[-1] != T:
                # Ensure last point is T for consistency
                if t_grid[-1] < T:
                    t_grid = np.concatenate([t_grid, [T]])
                else:
                    t_grid[-1] = T

        # Simulate short rate paths
        rates = self.model.simulate_short_rate(
            t_grid=t_grid, n_paths=self.n_paths, seed=seed
        )

        # Calculate discount factors along paths using variable step sizes
        dt = np.diff(t_grid)  # shape (n_steps,)
        # cumulative integral of r dt across steps
        cumulative_rates = np.cumsum(rates[:, :-1] * dt[None, :], axis=1)
        discount_factors = np.exp(-cumulative_rates)
        discount_factors = np.column_stack([np.ones(self.n_paths), discount_factors])

        paths = {"rates": rates, "times": t_grid, "discount_factors": discount_factors}

        if use_cache:
            self.paths_cache = paths

        return paths

    def get_path_statistics(self, paths: Dict[str, np.ndarray]) -> Dict:
        """Calculate basic statistics for validation"""
        rates = paths["rates"]

        return {
            "mean_terminal_rate": np.mean(rates[:, -1]),
            "std_terminal_rate": np.std(rates[:, -1]),
            "mean_path": np.mean(rates, axis=0),
            "percentiles": np.percentile(rates[:, -1], [5, 50, 95]),
        }
