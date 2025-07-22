# Technical Implementation Plan: Longstaff-Schwartz Option Pricing Engine

## Overview

This document outlines the technical implementation of a Longstaff-Schwartz (LS) option pricing engine with Hull-White interest rate modeling. The focus is on building a minimal, production-ready system without over-engineering.

## Core Dependencies

```python
numpy==1.24.3          # Numerical computing
pandas==2.0.3          # Data handling  
scipy==1.11.1          # Interpolation and optimization
scikit-learn==1.3.0    # Regression models
statsmodels==0.14.0    # Statistical analysis
matplotlib==3.7.2      # Visualization
numba==0.57.1          # Performance optimization
```

## Project Structure

```
ls_pricing/
├── core/
│   ├── __init__.py
│   ├── hull_white.py      # Hull-White model implementation
│   └── curves.py          # Yield curve handling
├── engine/
│   ├── __init__.py
│   ├── monte_carlo.py     # Path generation
│   └── longstaff_schwartz.py  # LS algorithm
├── utils/
│   ├── __init__.py
│   └── interpolation.py   # Helper functions
├── tests/
│   └── test_pricing.py    # Unit tests
└── examples/
    └── simple_option.py   # Usage examples
```

## Phase 1: Core Components (Week 1)

### 1.1 Yield Curve Infrastructure

```python
# core/curves.py

import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class YieldCurve:
    """Simple yield curve container"""
    tenors: np.ndarray  # In years
    rates: np.ndarray   # Decimal format
    curve_date: str
    
    def __post_init__(self):
        # Build interpolator on initialization
        self.interpolator = CubicSpline(self.tenors, self.rates, 
                                       extrapolate=True)
    
    def get_rate(self, t: float) -> float:
        """Get interpolated rate at time t"""
        return float(self.interpolator(t))
    
    def get_discount_factor(self, t: float) -> float:
        """Get discount factor P(0,t)"""
        rate = self.get_rate(t)
        return np.exp(-rate * t)
    
    def get_forward_rate(self, t1: float, t2: float) -> float:
        """Get forward rate F(t1,t2)"""
        df1 = self.get_discount_factor(t1)
        df2 = self.get_discount_factor(t2)
        return -np.log(df2/df1) / (t2 - t1)
```

### 1.2 Hull-White Model

```python
# core/hull_white.py

import numpy as np
from numba import jit

class HullWhiteModel:
    """
    One-factor Hull-White model
    dr(t) = [theta(t) - a*r(t)]dt + sigma*dW(t)
    """
    
    def __init__(self, a: float, sigma: float, yield_curve: YieldCurve):
        self.a = a
        self.sigma = sigma
        self.yield_curve = yield_curve
        
        # Pre-calculate theta function for efficiency
        self._calibrate_theta()
        
    def _calibrate_theta(self):
        """Calibrate theta(t) to match initial yield curve"""
        # For Hull-White, theta(t) has analytical form:
        # theta(t) = df/dt + a*f(t) + sigma^2/(2a)*(1-exp(-2at))
        # where f(t) is instantaneous forward rate
        
        # We'll pre-compute theta at simulation time points
        self.theta_grid = {}
        
    def simulate_paths(self, n_paths: int, n_steps: int, T: float, 
                       seed: int = None) -> Dict[str, np.ndarray]:
        """
        Generate interest rate paths
        Returns dict with 'rates', 'times', 'discount_factors'
        """
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Generate random shocks (antithetic for variance reduction)
        n_half = n_paths // 2
        dW = np.sqrt(dt) * np.random.randn(n_half, n_steps)
        dW = np.vstack([dW, -dW])  # Antithetic paths
        
        # Initialize rate paths
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.yield_curve.get_rate(0)  # Start at short rate
        
        # Simulate using Euler scheme (could upgrade to exact later)
        rates = self._simulate_euler(rates, times, dW, dt)
        
        # Calculate discount factors along paths
        discount_factors = self._calculate_path_discounts(rates, dt)
        
        return {
            'rates': rates,
            'times': times,
            'discount_factors': discount_factors
        }
    
    @staticmethod
    @jit(nopython=True)
    def _simulate_euler(rates, times, dW, dt):
        """Numba-optimized Euler simulation"""
        n_paths, n_steps_plus = rates.shape
        a, sigma = 0.1, 0.01  # Pass these properly in real implementation
        
        for i in range(n_paths):
            for j in range(n_steps_plus - 1):
                # Simple theta for now - enhance later
                theta = 0.05  
                drift = theta - a * rates[i, j]
                rates[i, j+1] = rates[i, j] + drift * dt + sigma * dW[i, j]
                
        return rates
    
    def _calculate_path_discounts(self, rates: np.ndarray, 
                                  dt: float) -> np.ndarray:
        """Calculate discount factors along each path"""
        # Simple approximation: D(t,T) ≈ exp(-∫r(s)ds)
        cumulative_rates = np.cumsum(rates[:, :-1] * dt, axis=1)
        discounts = np.exp(-cumulative_rates)
        return np.column_stack([np.ones(rates.shape[0]), discounts])
```

## Phase 2: Monte Carlo Engine (Week 2)

### 2.1 Monte Carlo Path Generator

```python
# engine/monte_carlo.py

import numpy as np
from typing import Dict, Optional
from core.hull_white import HullWhiteModel

class MonteCarloEngine:
    """Wrapper for Monte Carlo simulation with diagnostics"""
    
    def __init__(self, model: HullWhiteModel, n_paths: int = 10000, 
                 n_steps: int = 100):
        self.model = model
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.paths_cache = None
        
    def generate_paths(self, T: float, seed: Optional[int] = None,
                      use_cache: bool = False) -> Dict[str, np.ndarray]:
        """Generate paths with optional caching"""
        
        if use_cache and self.paths_cache is not None:
            return self.paths_cache
            
        paths = self.model.simulate_paths(
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            T=T,
            seed=seed
        )
        
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
```

## Phase 3: Longstaff-Schwartz Implementation (Week 2-3)

### 3.1 Regression Engine

```python
# engine/longstaff_schwartz.py

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Callable, Optional

class RegressionBasis:
    """Handle basis function generation"""
    
    @staticmethod
    def laguerre_polynomials(x: np.ndarray, degree: int = 3) -> np.ndarray:
        """Generate Laguerre polynomial basis"""
        n = len(x)
        basis = np.zeros((n, degree + 1))
        
        # L0 = 1, L1 = 1-x, L2 = 1-2x+x²/2, etc.
        basis[:, 0] = 1
        if degree >= 1:
            basis[:, 1] = 1 - x
        if degree >= 2:
            basis[:, 2] = 1 - 2*x + x**2/2
        if degree >= 3:
            basis[:, 3] = 1 - 3*x + 3*x**2/2 - x**3/6
            
        return basis
    
    @staticmethod
    def polynomial_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
        """Simple polynomial basis using sklearn"""
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        return poly.fit_transform(x.reshape(-1, 1))
```

### 3.2 Main Longstaff-Schwartz Engine

```python
class LongstaffSchwartzEngine:
    """
    Main LS implementation
    Keep it simple - focus on American put/call first
    """
    
    def __init__(self, mc_engine: MonteCarloEngine,
                 basis_type: str = 'laguerre',
                 basis_degree: int = 3,
                 regression_type: str = 'ols'):
        
        self.mc_engine = mc_engine
        self.basis_type = basis_type
        self.basis_degree = basis_degree
        
        # Set up regression model
        if regression_type == 'ols':
            self.regressor = LinearRegression(fit_intercept=False)
        else:  # ridge
            self.regressor = Ridge(alpha=1e-6, fit_intercept=False)
            
        # Set up basis function
        if basis_type == 'laguerre':
            self.basis_func = RegressionBasis.laguerre_polynomials
        else:
            self.basis_func = RegressionBasis.polynomial_basis
    
    def price_american_option(self, strike: float, maturity: float,
                            option_type: str = 'put',
                            exercise_dates: Optional[List[float]] = None) -> Dict:
        """
        Price American option using LS method
        
        Returns:
            Dict with 'price', 'std_error', 'exercise_boundary'
        """
        
        # Generate paths
        paths = self.mc_engine.generate_paths(T=maturity)
        rates = paths['rates']
        times = paths['times']
        discount_factors = paths['discount_factors']
        
        n_paths, n_steps = rates.shape[0], rates.shape[1] - 1
        
        # Set exercise dates (default: all time steps)
        if exercise_dates is None:
            exercise_steps = list(range(1, n_steps + 1))
        else:
            # Map dates to step indices
            exercise_steps = [np.argmin(np.abs(times - t)) for t in exercise_dates]
        
        # Initialize cash flow matrix
        cash_flows = np.zeros((n_paths, n_steps + 1))
        exercise_boundary = {}
        
        # Terminal payoff
        terminal_payoff = self._calculate_payoff(rates[:, -1], strike, option_type)
        cash_flows[:, -1] = terminal_payoff
        
        # Backward induction
        for i in range(len(exercise_steps) - 1, 0, -1):
            step = exercise_steps[i]
            
            # Current state (using rates as state variable)
            X = rates[:, step]
            
            # Immediate exercise value
            immediate_value = self._calculate_payoff(X, strike, option_type)
            
            # Only regress on ITM paths
            itm_mask = immediate_value > 0
            
            if np.sum(itm_mask) > 10:  # Need enough points for regression
                # Future cash flows (discounted back to current step)
                future_cf = self._discount_future_cashflows(
                    cash_flows[:, step+1:], 
                    discount_factors[:, step:],
                    step
                )
                
                # Regression on ITM paths
                X_itm = X[itm_mask].reshape(-1, 1)
                y_itm = future_cf[itm_mask]
                
                # Generate basis functions
                basis = self.basis_func(X_itm.flatten(), self.basis_degree)
                
                # Fit regression
                self.regressor.fit(basis, y_itm)
                
                # Predict continuation value for all paths
                basis_all = self.basis_func(X.flatten(), self.basis_degree)
                continuation_value = np.zeros(n_paths)
                continuation_value[itm_mask] = self.regressor.predict(
                    basis_all[itm_mask]
                )
                
                # Exercise decision
                exercise = (immediate_value > continuation_value) & itm_mask
                
                # Update cash flows
                cash_flows[exercise, step] = immediate_value[exercise]
                cash_flows[exercise, step+1:] = 0
                
                # Store exercise boundary (for analysis)
                if np.sum(exercise) > 0:
                    exercise_boundary[times[step]] = {
                        'threshold_rate': np.mean(X[exercise]),
                        'exercise_prob': np.mean(exercise)
                    }
        
        # Calculate option value
        # For each path, take first non-zero cash flow and discount to present
        option_values = self._calculate_option_values(cash_flows, discount_factors)
        
        return {
            'price': np.mean(option_values),
            'std_error': np.std(option_values) / np.sqrt(n_paths),
            'exercise_boundary': exercise_boundary,
            'paths_used': n_paths
        }
    
    def _calculate_payoff(self, spot: np.ndarray, strike: float,
                         option_type: str) -> np.ndarray:
        """Calculate option payoff"""
        if option_type == 'put':
            return np.maximum(strike - spot, 0)
        else:  # call
            return np.maximum(spot - strike, 0)
    
    def _discount_future_cashflows(self, future_cf: np.ndarray,
                                  discount_factors: np.ndarray,
                                  current_step: int) -> np.ndarray:
        """Discount future cash flows to current step"""
        # Sum of discounted future cash flows
        result = np.zeros(future_cf.shape[0])
        
        for i in range(future_cf.shape[1]):
            if i < discount_factors.shape[1]:
                result += future_cf[:, i] * discount_factors[:, i]
                
        return result
    
    def _calculate_option_values(self, cash_flows: np.ndarray,
                               discount_factors: np.ndarray) -> np.ndarray:
        """Calculate present value for each path"""
        pv = np.zeros(cash_flows.shape[0])
        
        for i in range(cash_flows.shape[0]):
            # Find first non-zero cash flow
            cf_indices = np.where(cash_flows[i] > 0)[0]
            if len(cf_indices) > 0:
                first_cf_idx = cf_indices[0]
                pv[i] = cash_flows[i, first_cf_idx] * discount_factors[i, first_cf_idx-1]
                
        return pv
```

## Phase 4: Calibration (Week 3)

### 4.1 Hull-White Calibration to Swaptions

```python
# core/calibration.py

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List

class HullWhiteCalibrator:
    """Calibrate Hull-White to swaption volatilities"""
    
    def __init__(self, yield_curve: YieldCurve):
        self.yield_curve = yield_curve
        
    def calibrate_to_swaptions(self, swaption_data: Dict) -> Dict[str, float]:
        """
        Calibrate a and sigma to match swaption prices
        
        swaption_data format:
        {
            'expiries': [1, 2, 3, 5],      # Years
            'tenors': [1, 2, 5, 10],        # Years  
            'vols': [[...], [...], ...]     # ATM vols
        }
        """
        
        # Initial guess
        x0 = [0.01, 0.01]  # [a, sigma]
        
        # Bounds: a in (0, 0.5), sigma in (0, 0.1)
        bounds = [(1e-4, 0.5), (1e-4, 0.1)]
        
        # Objective: minimize squared vol errors
        result = minimize(
            fun=self._calibration_objective,
            x0=x0,
            args=(swaption_data,),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        return {
            'a': result.x[0],
            'sigma': result.x[1],
            'success': result.success,
            'error': result.fun
        }
    
    def _calibration_objective(self, params: np.ndarray, 
                              swaption_data: Dict) -> float:
        """Calculate sum of squared vol errors"""
        a, sigma = params
        
        # For each swaption, calculate model implied vol
        # Compare to market vol
        # This is simplified - real implementation needs 
        # swaption pricer under Hull-White
        
        total_error = 0.0
        # ... implementation details ...
        
        return total_error
```

## Phase 5: Testing Framework (Week 4)

### 5.1 Unit Tests

```python
# tests/test_pricing.py

import numpy as np
import pytest
from core.curves import YieldCurve
from core.hull_white import HullWhiteModel
from engine.monte_carlo import MonteCarloEngine
from engine.longstaff_schwartz import LongstaffSchwartzEngine

class TestLongstaffSchwartz:
    
    def setup_method(self):
        """Set up test fixtures"""
        # Simple flat yield curve for testing
        tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
        rates = np.array([0.05] * len(tenors))
        self.yield_curve = YieldCurve(tenors, rates, '2024-01-01')
        
        # Hull-White model
        self.hw_model = HullWhiteModel(a=0.1, sigma=0.01, 
                                      yield_curve=self.yield_curve)
        
    def test_european_put_benchmark(self):
        """Test against Black-Scholes for European option"""
        mc_engine = MonteCarloEngine(self.hw_model, n_paths=50000)
        ls_engine = LongstaffSchwartzEngine(mc_engine)
        
        # Price European put (no early exercise)
        result = ls_engine.price_american_option(
            strike=0.05,
            maturity=1.0,
            option_type='put',
            exercise_dates=[1.0]  # Only at maturity
        )
        
        # Should match Black-Scholes within tolerance
        assert abs(result['price'] - 0.001) < 0.0001
        
    def test_american_put_early_exercise(self):
        """Test that American value >= European value"""
        mc_engine = MonteCarloEngine(self.hw_model, n_paths=10000)
        ls_engine = LongstaffSchwartzEngine(mc_engine)
        
        # American put with many exercise dates
        american_result = ls_engine.price_american_option(
            strike=0.05,
            maturity=1.0,
            option_type='put'
        )
        
        # European put
        european_result = ls_engine.price_american_option(
            strike=0.05,
            maturity=1.0,
            option_type='put',
            exercise_dates=[1.0]
        )
        
        assert american_result['price'] >= european_result['price']
```

## Performance Optimization

### Key Optimizations

1. **Numba JIT Compilation**
   - Apply to simulation loops
   - Apply to payoff calculations
   - ~10x speedup for core loops

2. **Vectorization**
   - Use numpy broadcasting
   - Avoid Python loops
   - Batch regression predictions

3. **Memory Management**
   - Pre-allocate arrays
   - Reuse path storage
   - Clear large arrays after use

### Performance Targets

- 50,000 paths in < 5 seconds
- Full calibration in < 30 seconds
- Price + Greeks in < 10 seconds

## Deployment Checklist

- [ ] Unit tests passing (>90% coverage)
- [ ] Performance benchmarks met
- [ ] Calibration validates against market
- [ ] Documentation complete
- [ ] Error handling for edge cases
- [ ] Logging for production debugging

## Next Steps

1. **Week 1**: Implement core Hull-White and curves
2. **Week 2**: Complete Monte Carlo and basic LS
3. **Week 3**: Add calibration and enhance LS
4. **Week 4**: Testing, optimization, and documentation
5. **Week 5**: Integration with existing systems

## Appendix: Quick Start Example

```python
# Example usage
from core.curves import YieldCurve
from core.hull_white import HullWhiteModel
from engine.monte_carlo import MonteCarloEngine
from engine.longstaff_schwartz import LongstaffSchwartzEngine

# Load market data
yield_curve = YieldCurve(
    tenors=np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10]),
    rates=np.array([0.045, 0.046, 0.048, 0.050, 0.052, 0.054, 0.055, 0.056]),
    curve_date='2024-01-15'
)

# Create and calibrate model
hw_model = HullWhiteModel(a=0.05, sigma=0.01, yield_curve=yield_curve)

# Set up engines
mc_engine = MonteCarloEngine(hw_model, n_paths=50000, n_steps=50)
ls_engine = LongstaffSchwartzEngine(mc_engine)

# Price option
result = ls_engine.price_american_option(
    strike=0.05,
    maturity=2.0,
    option_type='put'
)

print(f"Option Price: ${result['price']:.4f}")
print(f"Std Error: ${result['std_error']:.4f}")
```