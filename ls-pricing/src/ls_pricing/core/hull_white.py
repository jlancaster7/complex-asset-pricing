import numpy as np
from typing import Tuple, Optional
from .curves import YieldCurve


class HullWhiteModel:
    def __init__(self, a: float, sigma: float, yield_curve: YieldCurve):
        self.a = a
        self.sigma = sigma
        self.yield_curve = yield_curve
        # Cache for spline derivatives (initialized lazily)
        self._spot_deriv1 = None
        self._spot_deriv2 = None
        if a <= 0:
            raise ValueError("Mean reversion parameter 'a' must be positive")
        if sigma <= 0:
            raise ValueError("Volatility parameter 'sigma' must be positive")

    def simulate_short_rate(
        self, t_grid: np.ndarray, n_paths: int, seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        n_steps = len(t_grid)
        dt = np.diff(t_grid)
        r = np.zeros((n_paths, n_steps))
        r[:, 0] = self.yield_curve.get_rate(0.0)
        dW = np.random.randn(n_paths, n_steps - 1) * np.sqrt(dt)
        # Precompute theta over grid (left-point for intervals)
        theta_vals = self._theta_vectorized(t_grid[:-1])
        for i in range(1, n_steps):
            theta_t = theta_vals[i - 1]
            dr = (theta_t - self.a * r[:, i - 1]) * dt[i - 1] + self.sigma * dW[
                :, i - 1
            ]
            r[:, i] = r[:, i - 1] + dr
        return r

    def _prepare_spot_derivatives(self):
        """Prepare first and second derivatives of spot (zero) rate spline.
        Spot curve R(T) given by self.yield_curve._interpolator; use its derivatives.
        """
        if self._spot_deriv1 is None or self._spot_deriv2 is None:
            spline = self.yield_curve._interpolator
            self._spot_deriv1 = spline.derivative(1)
            self._spot_deriv2 = spline.derivative(2)

    def _theta_vectorized(self, t: np.ndarray) -> np.ndarray:
        """Analytic theta(t) using R(t) + t R'(t) formulation.
        f(0,t) = R(t) + t R'(t);  d/dt f(0,t) = 2 R'(t) + t R"(t)
        theta(t) = d/dt f(0,t) + a f(0,t) + (sigma^2/(2a))(1 - e^{-2 a t})
        Handle t=0 separately (limit as t→0: f(0,0)=R(0)).
        """
        t = np.asarray(t)
        self._prepare_spot_derivatives()
        spline = self.yield_curve._interpolator
        # Type assumptions after preparation
        assert self._spot_deriv1 is not None and self._spot_deriv2 is not None
        R = spline(t)
        R_p = self._spot_deriv1(t)  # type: ignore[operator]
        R_pp = self._spot_deriv2(t)  # type: ignore[operator]
        f0t = R + t * R_p
        df_dt = 2 * R_p + t * R_pp
        theta = (
            df_dt
            + self.a * f0t
            + (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * t))
        )
        # t=0 safe handling
        if np.any(t == 0):
            idx = t == 0
            R0 = spline(0.0)
            R0_p = self._spot_deriv1(0.0)  # type: ignore[operator]
            # f(0,0)=R0 ; df_dt at 0 = 2 R'(0)
            theta[idx] = (2 * R0_p) + self.a * R0
        return theta

    def _theta(self, t: float) -> float:
        # Retain single-point interface (used nowhere after vectorization but kept for API)
        return float(self._theta_vectorized(np.array([t]))[0])

    def zero_bond_price(self, t: float, T: float, r_t: float) -> float:
        if T == t:
            return 1.0
        B = (1 - np.exp(-self.a * (T - t))) / self.a
        P_0_T = self.yield_curve.get_discount_factor(T)
        P_0_t = self.yield_curve.get_discount_factor(t)
        if t == 0:
            f_0_t = self.yield_curve.get_rate(0.0)
        else:
            f_0_t = self.yield_curve.get_forward_rate(0, t)
        ln_A = (
            np.log(P_0_T / P_0_t)
            + B * f_0_t
            - (self.sigma**2 / (4 * self.a)) * B**2 * (1 - np.exp(-2 * self.a * t))
        )
        return np.exp(ln_A - B * r_t)

    def calibrate_to_swaptions(
        self, swaption_data: dict, initial_params: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        import warnings

        if initial_params is None:
            initial_params = (self.a, self.sigma)
        warnings.warn(
            "Hull-White calibration to swaptions is not implemented yet; returning initial parameters.",
            category=UserWarning,
        )
        return initial_params
