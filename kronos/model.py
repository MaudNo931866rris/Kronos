"""Core Kronos prediction model.

Implements the time-series forecasting logic used across example scripts.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KronosConfig:
    """Configuration for a Kronos prediction run."""
    window_size: int = 20          # look-back window (trading days)
    forecast_horizon: int = 5      # days ahead to forecast
    alpha: float = 0.3             # exponential smoothing factor
    trend_weight: float = 0.5      # weight given to linear trend vs EMA
    volatility_scale: float = 1.0  # multiplier for confidence interval width
    random_seed: Optional[int] = 42


@dataclass
class PredictionResult:
    """Output of a single Kronos forecast."""
    dates: list                    # forecast date labels (strings or ints)
    predicted: np.ndarray          # point forecast
    lower: np.ndarray              # lower confidence bound
    upper: np.ndarray              # upper confidence bound
    last_close: float              # last observed close price
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "dates": self.dates,
            "predicted": self.predicted.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "last_close": self.last_close,
            "metadata": self.metadata,
        }


class KronosModel:
    """Simple trend + EMA forecasting model.

    This is a lightweight baseline model that combines an exponential
    moving average with a linear trend estimate to produce short-horizon
    forecasts with symmetric confidence intervals.
    """

    def __init__(self, config: Optional[KronosConfig] = None):
        self.config = config or KronosConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_predict(self, close_prices: np.ndarray,
                    forecast_dates: Optional[list] = None) -> PredictionResult:
        """Fit on *close_prices* and return a forecast.

        Parameters
        ----------
        close_prices:
            1-D array of historical closing prices (oldest → newest).
        forecast_dates:
            Optional list of labels for the forecast steps.  If *None*,
            integer offsets 1..horizon are used.
        """
        prices = np.asarray(close_prices, dtype=float)
        if len(prices) < 2:
            raise ValueError("Need at least 2 data points to forecast.")

        window = min(self.config.window_size, len(prices))
        recent = prices[-window:]

        ema = self._ema(recent, self.config.alpha)
        trend = self._linear_trend(recent)
        sigma = float(np.std(np.diff(recent)))

        horizon = self.config.forecast_horizon
        predicted = np.empty(horizon)
        for h in range(1, horizon + 1):
            predicted[h - 1] = (
                self.config.trend_weight * (ema + trend * h)
                + (1 - self.config.trend_weight) * ema
            )

        half_width = (
            sigma
            * self.config.volatility_scale
            * np.sqrt(np.arange(1, horizon + 1))
        )
        lower = predicted - half_width
        upper = predicted + half_width

        if forecast_dates is None:
            forecast_dates = list(range(1, horizon + 1))

        return PredictionResult(
            dates=forecast_dates[:horizon],
            predicted=predicted,
            lower=lower,
            upper=upper,
            last_close=float(prices[-1]),
            metadata={"window": window, "trend_per_day": trend, "ema": ema},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(series: np.ndarray, alpha: float) -> float:
        """Return the final EMA value of *series*."""
        ema = series[0]
        for v in series[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return float(ema)

    @staticmethod
    def _linear_trend(series: np.ndarray) -> float:
        """Estimate average daily change via least-squares slope."""
        x = np.arange(len(series), dtype=float)
        slope = np.polyfit(x, series, 1)[0]
        return float(slope)
