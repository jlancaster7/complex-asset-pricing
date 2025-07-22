import numpy as np
from typing import Tuple, Optional
from .curves import YieldCurve


class HullWhiteModel:
    def __init__(
        self,
        a: float,
        sigma: float,
        yield_curve: YieldCurve
    ):
        self.a = a
        self.sigma = sigma
        self.yield_curve = yield_curve
        
        if a <= 0:
            raise ValueError("Mean reversion parameter 'a' must be positive")
        if sigma <= 0:
            raise ValueError("Volatility parameter 'sigma' must be positive")
    
    def simulate_short_rate(
        self,
        t_grid: np.ndarray,
        n_paths: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        
        n_steps = len(t_grid)
        dt = np.diff(t_grid)
        
        r = np.zeros((n_paths, n_steps))
        r[:, 0] = self.yield_curve.get_rate(0.0)
        
        dW = np.random.randn(n_paths, n_steps - 1) * np.sqrt(dt)
        
        for i in range(1, n_steps):
            theta_t = self._theta(t_grid[i-1])
            dr = (theta_t - self.a * r[:, i-1]) * dt[i-1] + self.sigma * dW[:, i-1]
            r[:, i] = r[:, i-1] + dr
        
        return r
    
    def _theta(self, t: float) -> float:
        dt = 1e-6
        f_t = self.yield_curve.get_forward_rate(t, t + dt)
        f_t_plus = self.yield_curve.get_forward_rate(t + dt, t + 2*dt)
        df_dt = (f_t_plus - f_t) / dt
        
        return df_dt + self.a * f_t + (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * t))
    
    def zero_bond_price(
        self,
        t: float,
        T: float,
        r_t: float
    ) -> float:
        if T == t:
            return 1.0
        
        B = (1 - np.exp(-self.a * (T - t))) / self.a
        
        P_0_T = self.yield_curve.get_discount_factor(T)
        P_0_t = self.yield_curve.get_discount_factor(t)
        
        if t == 0:
            f_0_t = self.yield_curve.get_rate(0.0)
        else:
            f_0_t = self.yield_curve.get_forward_rate(0, t)
        
        ln_A = np.log(P_0_T / P_0_t) + B * f_0_t - \
               (self.sigma**2 / (4 * self.a)) * B**2 * (1 - np.exp(-2 * self.a * t))
        
        return np.exp(ln_A - B * r_t)
    
    def calibrate_to_swaptions(
        self,
        swaption_data: dict,
        initial_params: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        if initial_params is None:
            initial_params = (self.a, self.sigma)
        
        return initial_params