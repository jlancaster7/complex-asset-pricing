import numpy as np
from datetime import datetime
from typing import Union, Optional
from scipy.interpolate import CubicSpline


class YieldCurve:
    def __init__(
        self,
        tenors: np.ndarray,
        rates: np.ndarray,
        curve_date: datetime,
        day_count: str = "ACT/365",
    ):
        self.tenors = np.asarray(tenors)
        self.rates = np.asarray(rates)
        self.day_count = day_count

        if len(self.tenors) != len(self.rates):
            raise ValueError("Tenors and rates must have the same length")

        if not np.all(np.diff(self.tenors) > 0):
            raise ValueError("Tenors must be strictly increasing")

        self._interpolator = CubicSpline(self.tenors, self.rates, extrapolate=True)

    def get_rate(self, tenor: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Handle both scalar and array inputs
        if isinstance(tenor, np.ndarray):
            # For array input, apply flat extrapolation element-wise
            result = np.zeros_like(tenor)
            for i, t in enumerate(tenor):
                if t <= self.tenors[0]:
                    result[i] = self.rates[0]
                elif t >= self.tenors[-1]:
                    result[i] = self.rates[-1]
                else:
                    result[i] = self._interpolator(t)
            return result
        else:
            # For scalar input
            if tenor <= self.tenors[0]:
                return self.rates[0]
            elif tenor >= self.tenors[-1]:
                return self.rates[-1]
            else:
                return float(self._interpolator(tenor))

    def get_discount_factor(
        self, tenor: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        rate = self.get_rate(tenor)
        return np.exp(-rate * tenor)

    def get_forward_rate(self, start_tenor: float, end_tenor: float) -> float:
        if start_tenor >= end_tenor:
            raise ValueError("Start tenor must be less than end tenor")

        df_start = self.get_discount_factor(start_tenor)
        df_end = self.get_discount_factor(end_tenor)

        return -np.log(df_end / df_start) / (end_tenor - start_tenor)

    def shift_curve(self, shift: float) -> "YieldCurve":
        return YieldCurve(
            self.tenors, self.rates + shift, self.curve_date, self.day_count
        )
