import numpy as np
from sklearn.linear_model import LinearRegression, Ridge  # type: ignore
from typing import Dict, List, Optional, Tuple
from .monte_carlo import MonteCarloEngine
from ..core.hull_white import HullWhiteModel


class RegressionBasis:
    """Handle basis function generation for LS regression"""
    
    @staticmethod
    def laguerre_polynomials(x: np.ndarray, degree: int = 3) -> np.ndarray:
        """
        Generate Laguerre polynomial basis
        L0 = 1
        L1 = 1 - x
        L2 = 1 - 2x + x²/2
        L3 = 1 - 3x + 3x²/2 - x³/6
        """
        n = len(x)
        basis = np.zeros((n, degree + 1))
        basis[:, 0] = 1.0
        if degree >= 1:
            basis[:, 1] = 1 - x
        if degree >= 2:
            basis[:, 2] = 1 - 2*x + x**2/2
        if degree >= 3:
            basis[:, 3] = 1 - 3*x + 3*x**2/2 - x**3/6
        return basis
    
    @staticmethod
    def polynomial_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
        """Simple polynomial basis: [1, x, x², x³, ...]"""
        n = len(x)
        basis = np.zeros((n, degree + 1))
        for i in range(degree + 1):
            basis[:, i] = x**i
        return basis


class LongstaffSchwartzEngine:
    """
    Main Longstaff-Schwartz implementation for American option pricing
    Focus on American options on zero bonds
    """
    
    def __init__(
        self,
        mc_engine: MonteCarloEngine,
        basis_type: str = 'laguerre',
        basis_degree: int = 3,
        regression_type: str = 'ols'
    ):
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
    
    def price_american_option(
        self,
        strike: float,
        option_maturity: float,
        bond_maturity: float,
        option_type: str = 'put',
        exercise_dates: Optional[List[float]] = None
    ) -> Dict:
        """
        Price American option on zero bond using LS method
        
        Args:
            strike: Strike price of the option
            option_maturity: Maturity of the option (T_option)
            bond_maturity: Maturity of the underlying bond (T_bond > T_option)
            option_type: 'put' or 'call'
            exercise_dates: List of exercise dates (if None, all time steps)
            
        Returns:
            Dict with 'price', 'std_error', 'exercise_boundary', 'exercise_prob'
        """
        if bond_maturity <= option_maturity:
            raise ValueError("Bond maturity must be greater than option maturity")
        paths = self.mc_engine.generate_paths(T=option_maturity)
        rates = paths['rates']
        times = paths['times']
        discount_factors = paths['discount_factors']
        n_paths, n_steps = rates.shape[0], rates.shape[1] - 1
        if exercise_dates is None:
            exercise_steps: List[int] = list(range(1, n_steps + 1))
        else:
            exercise_steps = [int(np.argmin(np.abs(times - t))) for t in exercise_dates]
            exercise_steps = [int(s) for s in exercise_steps if s > 0]
        cash_flows = np.zeros((n_paths, len(times)))
        exercise_time = np.zeros(n_paths)
        exercise_boundary: Dict[float, Dict[str, float]] = {}
        
        # Terminal payoff
        terminal_bond_prices = np.array([
            self.mc_engine.model.zero_bond_price(
                t=option_maturity,
                T=bond_maturity,
                r_t=rates[i, -1]
            ) for i in range(n_paths)
        ])
        
        terminal_payoff = self._calculate_payoff(
            terminal_bond_prices, strike, option_type
        )
        cash_flows[:, -1] = terminal_payoff
        exercise_time[terminal_payoff > 0] = option_maturity
        
        # Backward induction
        for i in range(len(exercise_steps) - 1, -1, -1):
            step = int(exercise_steps[i])
            current_time = times[step]
            
            # Current bond prices
            bond_prices = np.array([
                self.mc_engine.model.zero_bond_price(
                    t=current_time,
                    T=bond_maturity,
                    r_t=rates[j, step]
                ) for j in range(n_paths)
            ])
            
            # Immediate exercise value
            immediate_value = self._calculate_payoff(bond_prices, strike, option_type)
            
            # Only regress on ITM paths
            itm_mask = immediate_value > 0
            
            if np.sum(itm_mask) > 10:  # Need enough points for regression
                # Discount all future cash flows (step+1 .. end) back to current step using DF ratios
                future_cf_pv = self._discount_future_cashflows(cash_flows, discount_factors, int(step))
                
                # Regression on ITM paths using short rate as state variable
                X_itm = rates[itm_mask, step]
                y_itm = future_cf_pv[itm_mask]
                
                # Generate basis functions
                basis = self.basis_func(X_itm, self.basis_degree)
                
                # Fit regression
                self.regressor.fit(basis, y_itm)
                
                # Predict continuation value for ITM paths
                continuation_value = np.zeros(n_paths)
                continuation_value[itm_mask] = self.regressor.predict(basis)
                
                # Exercise decision: exercise if immediate > continuation
                exercise = (immediate_value > continuation_value) & itm_mask
                
                # Update cash flows
                cash_flows[exercise, step] = immediate_value[exercise]
                cash_flows[exercise, step+1:] = 0.0
                exercise_time[exercise] = current_time
                
                # Store exercise boundary info
                if np.sum(exercise) > 0:
                    exercise_boundary[current_time] = {
                        'mean_rate': float(np.mean(rates[exercise, step])),
                        'mean_bond_price': float(np.mean(bond_prices[exercise])),
                        'exercise_prob': float(np.mean(exercise)),
                        'n_exercised': float(np.sum(exercise))
                    }
        
        # Calculate option value
        option_values = self._calculate_option_values(cash_flows, discount_factors)
        
        # Calculate exercise probability
        exercise_prob = np.mean(exercise_time > 0)
        mean_exercise_time = np.mean(exercise_time[exercise_time > 0]) if exercise_prob > 0 else 0.0
        
        return {
            'price': float(np.mean(option_values)),
            'std_error': float(np.std(option_values) / np.sqrt(n_paths)),
            'exercise_boundary': exercise_boundary,
            'exercise_prob': float(exercise_prob),
            'mean_exercise_time': float(mean_exercise_time),
            'paths_used': n_paths
        }
    
    def _calculate_payoff(
        self,
        bond_prices: np.ndarray,
        strike: float,
        option_type: str
    ) -> np.ndarray:
        """Calculate option payoff"""
        if option_type.lower() == 'put':
            return np.maximum(strike - bond_prices, 0)
        return np.maximum(bond_prices - strike, 0)
    
    def _discount_future_cashflows(
        self,
        cash_flows: np.ndarray,
        discount_factors: np.ndarray,
        current_step: int
    ) -> np.ndarray:
        """Present value at current step of all future cash flows for each path.
        Uses pathwise discount factor ratios: PV_ti(CF_tj) = CF_tj * DF(0,tj)/DF(0,ti)."""
        df_current = discount_factors[:, current_step]
        result = np.zeros(cash_flows.shape[0])
        
        # Iterate over future steps only where any cash flow exists to avoid extra work
        future_indices = np.where(np.any(cash_flows[:, current_step+1:] > 0, axis=0))[0] + current_step + 1
        for j in future_indices:
            cf = cash_flows[:, j]
            if np.any(cf):
                result += cf * (discount_factors[:, j] / df_current)
                
        return result
    
    def _calculate_option_values(
        self,
        cash_flows: np.ndarray,
        discount_factors: np.ndarray
    ) -> np.ndarray:
        """Present value at time 0 for each path: first non-zero CF discounted by DF(0,t)."""
        pv = np.zeros(cash_flows.shape[0])
        
        for i in range(cash_flows.shape[0]):
            idx = np.argmax(cash_flows[i] > 0)
            if cash_flows[i, idx] > 0:  # valid exercise/terminal payoff
                pv[i] = cash_flows[i, idx] * discount_factors[i, idx]
                
        return pv
    
    def price_european_option(
        self,
        strike: float,
        option_maturity: float,
        bond_maturity: float,
        option_type: str = 'put'
    ) -> Dict:
        """
        Price European option for comparison
        Only exercises at maturity
        """
        if bond_maturity <= option_maturity:
            raise ValueError("Bond maturity must be greater than option maturity")
        
        # Generate paths
        paths = self.mc_engine.generate_paths(T=option_maturity)
        rates = paths['rates']
        discount_factors = paths['discount_factors']
        
        n_paths = rates.shape[0]
        
        # Terminal bond prices
        terminal_bond_prices = np.array([
            self.mc_engine.model.zero_bond_price(
                t=option_maturity,
                T=bond_maturity,
                r_t=rates[i, -1]
            ) for i in range(n_paths)
        ])
        
        # Terminal payoff
        payoffs = self._calculate_payoff(terminal_bond_prices, strike, option_type)
        
        option_values = payoffs * discount_factors[:, -1]
        
        return {
            'price': float(np.mean(option_values)),
            'std_error': float(np.std(option_values) / np.sqrt(n_paths)),
            'paths_used': n_paths
        }