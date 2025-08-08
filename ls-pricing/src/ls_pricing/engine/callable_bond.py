import numpy as np
from typing import Dict, Optional
from .longstaff_schwartz import LongstaffSchwartzEngine
from ..instruments.bonds import CallableBond


class CallableBondEngine(LongstaffSchwartzEngine):
    """
    Specialized Longstaff-Schwartz engine for pricing callable bonds

    Parameters
    ----------
    include_coupon_on_call : bool, default True
        If True, a call executed on a coupon date pays (call_price + coupon).
        If False, only call_price is paid (coupon excluded). Market conventions vary.
    """

    # Type annotations for caches
    _cached_paths: Optional[Dict[str, np.ndarray]]
    _cached_maturity: Optional[float]
    include_coupon_on_call: bool

    def __init__(
        self,
        mc_engine,
        basis_type: str = "laguerre",
        basis_degree: int = 3,
        regression_type: str = "ols",
        include_coupon_on_call: bool = True,
    ):
        super().__init__(mc_engine, basis_type, basis_degree, regression_type)
        self._cached_paths = None
        self._cached_maturity = None
        self.include_coupon_on_call = include_coupon_on_call

    def price_callable_bond(
        self, callable_bond: CallableBond, from_investor_perspective: bool = True
    ) -> Dict:
        """
        Price a callable bond using Longstaff-Schwartz method

        Args:
            callable_bond: CallableBond object
            from_investor_perspective: If True, return value to investor (who is short the call)
                                      If False, return value to issuer (who is long the call)

        Returns:
            Dict with:
                - callable_bond_price: Price of the callable bond
                - option_value: Value of the embedded call option
                - straight_bond_price: Price of non-callable bond
                - exercise_probability: Probability of call
                - mean_call_time: Expected call time (conditional on being called)
                - exercise_boundary: Exercise boundary information
        """
        # Use cached paths if available and maturity matches
        if (
            self._cached_paths is not None
            and self._cached_maturity == callable_bond.maturity
        ):
            paths = self._cached_paths
        else:
            # Generate new paths and cache them
            paths = self.mc_engine.generate_paths(T=callable_bond.maturity)
            self._cached_paths = paths
            self._cached_maturity = callable_bond.maturity

        rates = paths["rates"]
        times = paths["times"]
        discount_factors = paths.get("discount_factors", None)

        n_paths = rates.shape[0]

        # Get exercise times (only after first call date)
        exercise_indices = callable_bond.get_exercise_times(times)

        if len(exercise_indices) == 0:
            # Bond is never callable, return straight bond value
            straight_value = self._price_straight_bond(callable_bond)
            return {
                "callable_bond_price": straight_value,
                "option_value": 0.0,
                "straight_bond_price": straight_value,
                "exercise_probability": 0.0,
                "mean_call_time": 0.0,
                "exercise_boundary": {},
            }

        # Initialize cash flow tracking
        # For callable bond, we track when it's called and what the investor receives
        cash_flows = np.zeros((n_paths, len(times)))
        call_indicator = np.zeros(n_paths, dtype=bool)
        call_time = np.zeros(n_paths)
        exercise_boundary = {}
        # NEW: regression diagnostics per exercise (after regression smoothing)
        regression_diagnostics: Dict[float, Dict[str, object]] = {}

        # ADDED: Track issuer's value/liability at each node
        # This is key for making optimal call decisions
        issuer_values = np.zeros((n_paths, len(times)))

        # Calculate continuation values starting from maturity
        # At maturity, bond always pays final coupon + principal (never callable at maturity)
        terminal_idx = len(times) - 1
        issuer_values[:, terminal_idx] = -(
            callable_bond.coupon_payment + callable_bond.face_value
        )

        # Single backward induction loop through ALL time steps
        # This ensures continuation values properly reflect future optimal decisions
        for step in range(len(times) - 2, -1, -1):  # From second-to-last to first
            current_time = times[step]
            next_step = step + 1

            # First, propagate and discount issuer values from next period using DF ratios
            dt = times[next_step] - times[step]
            if discount_factors is None:
                # Fallback (should not happen): local step discount with short rate
                discount_rates = rates[:, step] + callable_bond.credit_spread
                step_df = np.exp(-discount_rates * dt)
            else:
                df_ratio = discount_factors[:, next_step] / discount_factors[:, step]
                spread_factor = np.exp(-callable_bond.credit_spread * dt)
                step_df = df_ratio * spread_factor
            issuer_values[:, step] = issuer_values[:, next_step] * step_df

            # Add coupon payment if this is a coupon date (negative for issuer)
            if self._is_coupon_date(current_time, callable_bond):
                issuer_values[:, step] -= callable_bond.coupon_payment

            # Now check if this is a callable date and make optimal exercise decision
            if step in exercise_indices:
                # Bond value at this node (what it's worth if held)
                bond_values = callable_bond.value_at_node(
                    self.mc_engine.model, current_time, rates[:, step]
                )

                # Immediate exercise value for issuer
                # Issuer saves: Bond Value - Call Price
                immediate_ex_value = bond_values - callable_bond.call_price

                # Only consider calling if bond is worth more than call price
                itm_mask = immediate_ex_value > 0

                if np.sum(itm_mask) > 10:  # Need enough ITM paths for regression
                    # The continuation value is already computed in issuer_values
                    continuation = issuer_values[:, step].copy()

                    # Use regression to smooth the continuation values
                    # This helps with the exercise decision
                    X_itm = rates[itm_mask, step]
                    y_itm = continuation[itm_mask]
                    basis = self.basis_func(X_itm, self.basis_degree)
                    self.regressor.fit(basis, y_itm)
                    # NEW: capture diagnostics for this regression
                    try:
                        y_fit = self.regressor.predict(basis)
                        mse = float(np.mean((y_fit - y_itm) ** 2))
                        # R^2 using model score (already centers appropriately for linear regression without intercept)
                        r2 = float(self.regressor.score(basis, y_itm))
                        coefs = (
                            self.regressor.coef_.tolist()
                            if hasattr(self.regressor, "coef_")
                            else []
                        )
                    except Exception:  # pragma: no cover - defensive
                        mse, r2, coefs = float("nan"), float("nan"), []
                    regression_diagnostics[current_time] = {
                        "coefficients": coefs,
                        "r2": r2,
                        "mse": mse,
                        "n_itm": int(np.sum(itm_mask)),
                        "n_paths": int(n_paths),
                        "basis_type": self.basis_type,
                        "basis_degree": self.basis_degree,
                    }

                    # Predict continuation value for ALL paths using the regression
                    # The regression was fitted on ITM paths but can predict for all
                    X_all = rates[:, step]
                    basis_all = self.basis_func(X_all, self.basis_degree)
                    continuation_value_smooth = self.regressor.predict(basis_all)

                    # Exercise decision from issuer perspective:
                    # Call if: cost of calling now < expected future cost
                    immediate_issuer_value = -callable_bond.call_price

                    # If calling on a coupon date, must also pay the coupon
                    if self.include_coupon_on_call and self._is_coupon_date(
                        current_time, callable_bond
                    ):
                        immediate_issuer_value -= callable_bond.coupon_payment

                    exercise = (
                        immediate_issuer_value > continuation_value_smooth
                    ) & itm_mask

                    # Update issuer values based on exercise decision
                    # This is crucial: future time steps will see this decision!
                    issuer_values[exercise, step] = immediate_issuer_value

                    # Update tracking arrays
                    call_indicator[exercise] = True
                    call_time[exercise] = current_time

                    # Record what investor receives when bond is called
                    call_payment = callable_bond.call_price
                    if self.include_coupon_on_call and self._is_coupon_date(
                        current_time, callable_bond
                    ):
                        call_payment += callable_bond.coupon_payment
                    cash_flows[exercise, step] = call_payment
                    # Zero out all future cash flows for called bonds
                    for future_step in range(step + 1, len(times)):
                        cash_flows[exercise, future_step] = 0

                    # Store exercise boundary
                    if np.sum(exercise) > 0:
                        exercise_boundary[current_time] = {
                            "mean_rate": np.mean(rates[exercise, step]),
                            "mean_bond_value": np.mean(bond_values[exercise]),
                            "call_probability": np.mean(exercise),
                            "n_called": np.sum(exercise),
                        }
                else:
                    # Not enough ITM paths for regression
                    # Use simple decision rule: call if deep in the money
                    immediate_issuer_value = -callable_bond.call_price

                    # If calling on a coupon date, must also pay the coupon
                    if self.include_coupon_on_call and self._is_coupon_date(
                        current_time, callable_bond
                    ):
                        immediate_issuer_value -= callable_bond.coupon_payment

                    continuation = issuer_values[:, step]

                    # Call if immediate cost < continuation cost
                    exercise = (immediate_issuer_value > continuation) & (
                        immediate_ex_value > 5.0
                    )

                    if np.sum(exercise) > 0:
                        call_indicator[exercise] = True
                        call_time[exercise] = current_time
                        call_payment = callable_bond.call_price
                        if self.include_coupon_on_call and self._is_coupon_date(
                            current_time, callable_bond
                        ):
                            call_payment += callable_bond.coupon_payment
                        cash_flows[exercise, step] = call_payment
                        issuer_values[exercise, step] = immediate_issuer_value
                        # Zero out all future cash flows for called bonds
                        for future_step in range(step + 1, len(times)):
                            cash_flows[exercise, future_step] = 0

        # Add regular coupon payments for bonds that weren't called
        self._add_scheduled_payments(
            cash_flows, callable_bond, times, call_indicator, call_time
        )

        # The callable bond price from investor perspective is the negative of
        # the issuer's value at time 0 (what the issuer owes)
        callable_price = float(-np.mean(issuer_values[:, 0]))

        # Calculate straight bond value for comparison
        straight_bond_value = self._price_straight_bond(callable_bond)
        option_value = straight_bond_value - callable_price

        # Calculate call statistics
        call_prob = float(np.mean(call_indicator))
        mean_call_time = (
            float(np.mean(call_time[call_indicator])) if call_prob > 0 else 0.0
        )

        results = {
            "callable_bond_price": callable_price,
            "option_value": option_value,
            "straight_bond_price": straight_bond_value,
            "exercise_probability": call_prob,
            "mean_call_time": mean_call_time,
            "exercise_boundary": exercise_boundary,
            "paths_used": n_paths,
            "include_coupon_on_call": self.include_coupon_on_call,
            "regression_diagnostics": regression_diagnostics,
        }

        # Adjust perspective if needed
        if not from_investor_perspective:
            # From issuer's perspective, they are long the option
            results["option_value"] = -results["option_value"]

        return results

    def _calculate_remaining_payments(
        self, bond: CallableBond, from_time: float, rates: np.ndarray
    ) -> np.ndarray:
        """Calculate PV of remaining payments if bond is not called"""
        values = np.zeros(len(rates))

        payment_times, cash_flows = bond.get_cash_flow_schedule(from_time=from_time)

        for i, r in enumerate(rates):
            pv = 0.0
            for t, cf in zip(payment_times, cash_flows):
                if t > from_time:
                    p_t_T = self.mc_engine.model.zero_bond_price(
                        t=from_time, T=t, r_t=r
                    )
                    pv += cf * p_t_T
            values[i] = pv

        return values

    def _add_scheduled_payments(
        self,
        cash_flows: np.ndarray,
        bond: CallableBond,
        times: np.ndarray,
        call_indicator: np.ndarray,
        call_time: np.ndarray,
    ):
        """Add scheduled coupon payments to cash flow array"""
        payment_times, payment_amounts = bond.get_cash_flow_schedule()

        # For each payment time, find closest time in grid
        for pmt_time, pmt_amt in zip(payment_times, payment_amounts):
            # Find closest time index
            time_idx = np.argmin(np.abs(times - pmt_time))

            # Add payment for bonds not yet called at this payment time
            for i in range(cash_flows.shape[0]):
                # Only add payment if:
                # 1. Bond was never called, OR
                # 2. Bond was called AFTER this payment time
                if not call_indicator[i] or call_time[i] > pmt_time:
                    cash_flows[i, time_idx] += pmt_amt

    def _calculate_continuation_value(
        self,
        bond: CallableBond,
        current_time: float,
        rates: np.ndarray,
        current_step: int,
        times: np.ndarray,
        call_indicator: np.ndarray,
    ) -> np.ndarray:
        """Calculate continuation value for the issuer"""
        n_paths = len(rates)

        # For bonds already called, continuation value is 0
        result = np.zeros(n_paths)

        # For bonds not yet called, the continuation value is:
        # Expected future value of the call option
        # Simplified: assume if not called now, bond continues to next period
        # and we get the bond value minus what we'd have to pay

        # Get remaining time to maturity
        remaining_time = bond.maturity - current_time
        if remaining_time <= 0:
            return result

        # For simplicity, calculate the expected bond value at next exercise date
        # This is a simplified approach - more sophisticated methods would
        # use the regression to estimate future option values

        return result

    def _discount_cash_flows(
        self,
        cash_flows: np.ndarray,
        rates: np.ndarray,
        times: np.ndarray,
        bond: Optional[CallableBond] = None,
    ) -> np.ndarray:
        """Discount cash flows to present value"""
        pv = np.zeros(cash_flows.shape[0])

        for i in range(cash_flows.shape[0]):
            for j in range(cash_flows.shape[1]):
                if cash_flows[i, j] > 0:
                    # Discount from time j to 0
                    avg_rate = np.mean(rates[i, : j + 1])
                    # Add credit spread to the discount rate
                    if bond is not None:
                        discount_rate = avg_rate + bond.credit_spread
                    else:
                        discount_rate = avg_rate
                    discount = np.exp(-discount_rate * times[j])
                    pv[i] += cash_flows[i, j] * discount

        return pv

    def _price_straight_bond(self, bond: CallableBond) -> float:
        """Price the bond without call feature using path discount factors for consistency."""
        # Ensure paths exist for maturity
        if self._cached_paths is None or self._cached_maturity != bond.maturity:
            self.generate_paths(bond.maturity, seed=None)
        assert self._cached_paths is not None
        paths = self._cached_paths
        times = paths["times"]
        discount_factors = paths["discount_factors"]
        n_paths = discount_factors.shape[0]

        payment_times, cash_flows = bond.get_cash_flow_schedule(from_time=0.0)
        if len(payment_times) == 0:
            return 0.0

        pv_per_path = np.zeros(n_paths)
        for t, cf in zip(payment_times, cash_flows):
            # nearest time index on the grid
            idx = int(np.argmin(np.abs(times - t)))
            # risk-free DF(0,t) from path and spread adjustment
            spread_adj = np.exp(-bond.credit_spread * (times[idx] - 0.0))
            pv_per_path += cf * discount_factors[:, idx] * spread_adj

        return float(np.mean(pv_per_path))

    def _is_coupon_date(self, time: float, bond: CallableBond) -> bool:
        """Check if a given time is a coupon payment date"""
        if time == 0:
            return False
        periods = time * bond.payment_frequency
        # Check if we're at an integer number of periods
        return abs(periods - round(periods)) < 1e-10

    def generate_paths(self, maturity: float, seed: Optional[int] = None) -> None:
        """
        Generate and cache Monte Carlo paths for a given maturity.

        Args:
            maturity: Bond maturity in years
            seed: Random seed for reproducibility
        """
        self._cached_paths = self.mc_engine.generate_paths(T=maturity, seed=seed)
        self._cached_maturity = maturity

    def clear_cached_paths(self) -> None:
        """Clear cached Monte Carlo paths."""
        self._cached_paths = None
        self._cached_maturity = None

    def has_cached_paths(self, maturity: float) -> bool:
        """
        Check if we have cached paths for the given maturity.

        Args:
            maturity: Bond maturity to check

        Returns:
            True if we have cached paths for this maturity
        """
        return self._cached_paths is not None and self._cached_maturity == maturity
