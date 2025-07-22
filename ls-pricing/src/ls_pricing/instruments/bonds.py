import numpy as np
from typing import List, Tuple, Optional
from ..core.hull_white import HullWhiteModel


class CouponBond:
    """
    Represents a fixed-rate coupon bond
    """

    def __init__(
        self,
        face_value: float = 100.0,
        coupon_rate: float = 0.05,
        maturity: float = 30.0,
        payment_frequency: int = 2,  # Semi-annual payments
        credit_spread: float = 0.0,  # Spread over risk-free rate
    ):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.payment_frequency = payment_frequency
        self.credit_spread = credit_spread
        self.coupon_payment = face_value * coupon_rate / payment_frequency

    def get_cash_flow_schedule(
        self, from_time: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get payment times and amounts from a given time

        Returns:
            payment_times: Array of payment times
            cash_flows: Array of cash flow amounts
        """
        # Calculate payment times
        dt = 1.0 / self.payment_frequency
        payment_times = []
        cash_flows = []

        # First payment time after from_time
        if from_time == 0:
            first_payment = 1.0 / self.payment_frequency
        else:
            # Find next payment strictly after from_time
            periods_elapsed = from_time * self.payment_frequency
            next_period = np.floor(periods_elapsed) + 1
            first_payment = next_period / self.payment_frequency

        # Generate all payment times
        t = first_payment
        while t <= self.maturity + 1e-10:  # Small epsilon for numerical precision
            payment_times.append(t)
            if abs(t - self.maturity) < 1e-10:
                # Final payment includes principal
                cash_flows.append(self.coupon_payment + self.face_value)
            else:
                cash_flows.append(self.coupon_payment)
            t += dt

        return np.array(payment_times), np.array(cash_flows)

    def value(
        self, hw_model: HullWhiteModel, current_time: float, current_rate: float
    ) -> float:
        """
        Value the bond using Hull-White model with credit spread adjustment

        Args:
            hw_model: Hull-White model for discounting
            current_time: Current time
            current_rate: Current short rate

        Returns:
            Present value of the bond
        """
        payment_times, cash_flows = self.get_cash_flow_schedule(from_time=current_time)

        if len(payment_times) == 0:
            return 0.0
        
        # Calculate present value
        pv = 0.0
        for t, cf in zip(payment_times, cash_flows):
            if t > current_time:
                # Get risk-free discount factor from Hull-White
                p_rf = hw_model.zero_bond_price(t=current_time, T=t, r_t=current_rate)

                # Adjust for credit spread
                # P_credit = P_rf * exp(-spread * (T-t))
                p_credit = p_rf * np.exp(-self.credit_spread * (t - current_time))
                
                pv += cf * p_credit

        return pv

    def value_at_node(
        self, hw_model: HullWhiteModel, current_time: float, current_rates: np.ndarray
    ) -> np.ndarray:
        """
        Value the bond for multiple rate scenarios

        Args:
            hw_model: Hull-White model
            current_time: Current time
            current_rates: Array of short rates

        Returns:
            Array of bond values
        """
        return np.array([self.value(hw_model, current_time, r) for r in current_rates])

    def yield_to_maturity(self, price: float, current_time: float = 0.0) -> float:
        """
        Calculate yield to maturity given price
        Simple implementation using Newton-Raphson
        """
        payment_times, cash_flows = self.get_cash_flow_schedule(from_time=current_time)

        # Initial guess
        ytm = self.coupon_rate

        # Newton-Raphson iteration
        for _ in range(50):
            # Calculate price and derivative
            pv = 0.0
            dpv_dy = 0.0

            for t, cf in zip(payment_times, cash_flows):
                dt = t - current_time
                if dt > 0:
                    df = np.exp(-ytm * dt)
                    pv += cf * df
                    dpv_dy -= cf * dt * df

            # Update
            error = pv - price
            if abs(error) < 1e-8:
                break

            ytm -= error / dpv_dy

        return ytm


class CallableBond(CouponBond):
    """
    Callable bond with American call option
    """

    def __init__(
        self,
        face_value: float = 100.0,
        coupon_rate: float = 0.05,
        maturity: float = 30.0,
        payment_frequency: int = 2,
        first_call_date: float = 10.0,
        call_price: float = 100.0,  # Callable at par
        credit_spread: float = 0.0,  # Spread over risk-free rate
    ):
        super().__init__(
            face_value, coupon_rate, maturity, payment_frequency, credit_spread
        )
        self.first_call_date = first_call_date
        self.call_price = call_price

    def is_callable(self, current_time: float) -> bool:
        """Check if bond is currently callable"""
        return current_time >= self.first_call_date

    def get_exercise_times(self, time_grid: np.ndarray) -> List[int]:
        """
        Get indices in time grid where exercise is allowed

        Args:
            time_grid: Array of times

        Returns:
            List of indices where exercise is allowed
        """
        exercise_indices = []
        for i, t in enumerate(time_grid):
            if t >= self.first_call_date and t < self.maturity:
                exercise_indices.append(i)
        return exercise_indices
