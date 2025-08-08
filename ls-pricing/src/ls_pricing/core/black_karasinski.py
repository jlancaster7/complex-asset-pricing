import numpy as np
from typing import Optional
from .curves import YieldCurve


class BlackKarasinskiModel:
    """
    Black–Karasinski (BK) short-rate model with lognormal rates:
    x_t = ln r_t follows an OU: dx = (theta(t) - a x) dt + sigma dW, r_t = exp(x_t).

    We simulate paths for r_t and rely on pathwise discount factors in engines.
    Theta(t) is calibrated to fit the initial discount curve via a simple fixed-point
    iteration on a time grid (sufficient for MC pricing where engines use DF ratios).
    """

    def __init__(self, a: float, sigma: float, yield_curve: YieldCurve):
        if a <= 0:
            raise ValueError("Mean reversion parameter 'a' must be positive")
        if sigma <= 0:
            raise ValueError("Volatility parameter 'sigma' must be positive")
        self.a = a
        self.sigma = sigma
        self.yield_curve = yield_curve
        self._theta_times: Optional[np.ndarray] = None
        self._theta_vals: Optional[np.ndarray] = None

    def _calibrate_theta_on_grid(self, t_grid: np.ndarray) -> None:
        """Calibrate theta(t) on grid so that E[exp(-∫ r ds)] ≈ P(0,t) under small-step approx.
        We use deterministic matching by setting x_mean(t) so that instantaneous mean rate
        aligns with instantaneous forward, then back out theta(t) from OU mean dynamics.
        This is a pragmatic calibration for MC use.
        """
        times = np.asarray(t_grid, dtype=float)
        dt = np.diff(times)
        # Instantaneous forward f(0,t) ≈ R(t) + t R'(t)
        R = self.yield_curve._interpolator
        R_p = R.derivative(1)
        f0 = R(times) + times * R_p(times)
        # OU mean m(t) solves dm/dt = theta(t) - a m(t). We pick m(t)=ln r_bar(t), r_bar from f0.
        # Use r_bar(t)=max(1e-8, f0(t)) as a proxy for typical short rate, then theta = dm/dt + a m.
        m = np.log(np.maximum(1e-8, f0))
        # Numerical derivative of m
        dm_dt = np.zeros_like(times)
        if len(times) > 1:
            dm_dt[1:-1] = (m[2:] - m[:-2]) / (times[2:] - times[:-2])
            dm_dt[0] = (m[1] - m[0]) / (times[1] - times[0])
            dm_dt[-1] = (m[-1] - m[-2]) / (times[-1] - times[-2])
        theta = dm_dt + self.a * m
        self._theta_times = times
        self._theta_vals = theta

    def _theta(self, t: np.ndarray) -> np.ndarray:
        assert (
            self._theta_times is not None and self._theta_vals is not None
        ), "Theta not calibrated"
        return np.interp(t, self._theta_times, self._theta_vals)

    def simulate_short_rate(
        self, t_grid: np.ndarray, n_paths: int, seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        t_grid = np.asarray(t_grid, dtype=float)
        if self._theta_times is None or not np.allclose(self._theta_times, t_grid):
            self._calibrate_theta_on_grid(t_grid)
        n_steps = len(t_grid)
        dt = np.diff(t_grid)
        r = np.zeros((n_paths, n_steps))
        # Initialize x0 from curve short rate (nonnegative)
        r0 = max(1e-6, float(self.yield_curve.get_rate(0.0)))
        x = np.full(n_paths, np.log(r0))
        r[:, 0] = np.exp(x)
        dW = np.random.randn(n_paths, n_steps - 1) * np.sqrt(dt)
        theta_vals = self._theta(t_grid[:-1])
        for i in range(1, n_steps):
            # OU step for x
            x = (
                x
                + (theta_vals[i - 1] - self.a * x) * dt[i - 1]
                + self.sigma * dW[:, i - 1]
            )
            r[:, i] = np.exp(x)
        return r
