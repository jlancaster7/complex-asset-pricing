import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
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
        
        # L0
        basis[:, 0] = 1.0
        
        if degree >= 1:
            # L1
            basis[:, 1] = 1 - x
            
        if degree >= 2:
            # L2
            basis[:, 2] = 1 - 2*x + x**2/2
            
        if degree >= 3:
            # L3
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
        
        # Generate paths
        paths = self.mc_engine.generate_paths(T=option_maturity)
        rates = paths['rates']
        times = paths['times']
        discount_factors = paths['discount_factors']
        
        n_paths, n_steps = rates.shape[0], rates.shape[1] - 1
        
        # Set exercise dates (default: all time steps except t=0)
        if exercise_dates is None:
            exercise_steps = list(range(1, n_steps + 1))
        else:
            # Map exercise dates to step indices
            exercise_steps = [np.argmin(np.abs(times - t)) for t in exercise_dates]
            exercise_steps = [s for s in exercise_steps if s > 0]  # Exclude t=0
        
        # Initialize cash flow matrix
        cash_flows = np.zeros((n_paths, n_steps + 1))
        exercise_time = np.zeros(n_paths)  # Track when each path exercises
        exercise_boundary = {}
        
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
            step = exercise_steps[i]
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
                # Future cash flows (discounted back to current step)
                future_cf = self._discount_future_cashflows(
                    cash_flows[:, step+1:],
                    rates[:, step:],
                    times[step:],
                    step
                )
                
                # Regression on ITM paths using short rate as state variable
                X_itm = rates[itm_mask, step]
                y_itm = future_cf[itm_mask]
                
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
                cash_flows[exercise, step+1:] = 0
                exercise_time[exercise] = current_time
                
                # Store exercise boundary info
                if np.sum(exercise) > 0:
                    exercise_boundary[current_time] = {
                        'mean_rate': np.mean(rates[exercise, step]),
                        'mean_bond_price': np.mean(bond_prices[exercise]),
                        'exercise_prob': np.mean(exercise),
                        'n_exercised': np.sum(exercise)
                    }
        
        # Calculate option value
        option_values = self._calculate_option_values(
            cash_flows, rates, times
        )
        
        # Calculate exercise probability
        exercise_prob = np.mean(exercise_time > 0)
        mean_exercise_time = np.mean(exercise_time[exercise_time > 0]) if exercise_prob > 0 else 0
        
        return {
            'price': np.mean(option_values),
            'std_error': np.std(option_values) / np.sqrt(n_paths),
            'exercise_boundary': exercise_boundary,
            'exercise_prob': exercise_prob,
            'mean_exercise_time': mean_exercise_time,
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
        else:  # call
            return np.maximum(bond_prices - strike, 0)
    
    def _discount_future_cashflows(
        self,
        future_cf: np.ndarray,
        rates: np.ndarray,
        times: np.ndarray,
        current_step: int
    ) -> np.ndarray:
        """Discount future cash flows to current time"""
        result = np.zeros(future_cf.shape[0])
        
        # For each future time step
        for i in range(future_cf.shape[1]):
            if i < rates.shape[1] - 1:
                # Calculate discount factor from current time to future time
                dt = times[i+1] - times[0]
                # Use average rate for discounting
                avg_rates = np.mean(rates[:, :i+1], axis=1)
                discount = np.exp(-avg_rates * dt)
                result += future_cf[:, i] * discount
                
        return result
    
    def _calculate_option_values(
        self,
        cash_flows: np.ndarray,
        rates: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """Calculate present value of option for each path"""
        pv = np.zeros(cash_flows.shape[0])
        
        for i in range(cash_flows.shape[0]):
            # Find first non-zero cash flow (exercise time)
            cf_indices = np.where(cash_flows[i] > 0)[0]
            if len(cf_indices) > 0:
                exercise_idx = cf_indices[0]
                exercise_time = times[exercise_idx]
                
                # Discount from exercise time to present
                if exercise_time > 0:
                    # Average rate from 0 to exercise time
                    avg_rate = np.mean(rates[i, :exercise_idx+1])
                    discount = np.exp(-avg_rate * exercise_time)
                    pv[i] = cash_flows[i, exercise_idx] * discount
                else:
                    pv[i] = cash_flows[i, exercise_idx]
                    
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
        times = paths['times']
        
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
        
        # Discount to present
        avg_rates = np.mean(rates, axis=1)
        discount_factors = np.exp(-avg_rates * option_maturity)
        option_values = payoffs * discount_factors
        
        return {
            'price': np.mean(option_values),
            'std_error': np.std(option_values) / np.sqrt(n_paths),
            'paths_used': n_paths
        }