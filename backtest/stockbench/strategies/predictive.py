from __future__ import annotations

from typing import Callable, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from .rule_based import _BaseLongShortStrategy


DesiredState = Tuple[bool | None, str]


class _BasePredictiveStrategy(_BaseLongShortStrategy):
    strategy_name = "predictive"

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        self.params = self._predictive_cfg()
        self.return_threshold = float(self.params.get("return_threshold", 0.002) or 0.002)
        self.min_history = int(self.params.get("min_history", 30) or 30)
        self.lookback_window = int(self.params.get("lookback_window", 60) or 60)
        self._lookback_days = max(self.lookback_window, self.min_history)

    def _predictive_cfg(self) -> dict:
        return dict(
            ((self.cfg or {}).get("predictive_strategies", {}) or {}).get(
                self.strategy_name, {}
            )
            or {}
        )

    def _closes(self, ctx: Dict, symbol: str) -> pd.Series:
        history = self._history(ctx, symbol, self._lookback_days)
        closes = pd.to_numeric(history.get("close"), errors="coerce").dropna()
        if self.lookback_window > 0:
            closes = closes.tail(self.lookback_window)
        return closes.astype(float)

    def _signal_from_predicted_return(self, predicted_return: float, reason: str) -> DesiredState:
        if predicted_return > self.return_threshold:
            return True, f"{reason}|predicted_return={predicted_return:.6f}"
        if predicted_return < -self.return_threshold:
            return False, f"{reason}|predicted_return={predicted_return:.6f}"
        return None, f"{reason}|predicted_return={predicted_return:.6f}"


class ARIMAStrategy(_BasePredictiveStrategy):
    strategy_name = "arima"

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        order = self.params.get("order", [1, 1, 1]) or [1, 1, 1]
        if isinstance(order, str):
            order = [int(part.strip()) for part in order.split(",") if part.strip()]
        if len(order) != 3:
            order = [1, 1, 1]
        self.order = tuple(int(part) for part in order)
        self.min_history = max(self.min_history, max(self.order) + 8)

    def _desired_state(self, *, symbol, ctx, current_price, current_value) -> DesiredState:
        del current_value
        closes = self._closes(ctx, symbol)
        if len(closes) < self.min_history:
            return None, "arima_insufficient_history"
        if float(current_price or 0.0) <= 0.0:
            return None, "arima_invalid_open_price"
        if closes.nunique() <= 1:
            return None, "arima_flat_history"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = ARIMA(
                    closes,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            forecast = result.forecast(steps=1)
            forecast_price = float(forecast.iloc[0])
        except Exception as exc:
            return None, f"arima_fit_failed={type(exc).__name__}"
        predicted_return = (forecast_price - float(current_price)) / float(current_price)
        return self._signal_from_predicted_return(
            predicted_return,
            f"arima_forecast={forecast_price:.4f}",
        )


class XGBoostStrategy(_BasePredictiveStrategy):
    strategy_name = "xgboost"

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)
        self.n_lags = int(self.params.get("n_lags", 5) or 5)
        self.min_samples = int(self.params.get("min_samples", 20) or 20)
        self.n_estimators = int(self.params.get("n_estimators", 64) or 64)
        self.max_depth = int(self.params.get("max_depth", 3) or 3)
        self.learning_rate = float(self.params.get("learning_rate", 0.05) or 0.05)
        self.subsample = float(self.params.get("subsample", 1.0) or 1.0)
        self.colsample_bytree = float(self.params.get("colsample_bytree", 1.0) or 1.0)
        self.random_state = int(self.params.get("random_state", 42) or 42)
        self.min_history = max(self.min_history, self.n_lags + self.min_samples + 1)

    def _dataset(self, closes: pd.Series) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        returns = closes.pct_change().dropna().to_numpy(dtype=float)
        if len(returns) <= self.n_lags:
            return None, None
        features: list[np.ndarray] = []
        labels: list[float] = []
        for idx in range(self.n_lags, len(returns)):
            window = returns[idx - self.n_lags : idx]
            target = returns[idx]
            features.append(window)
            labels.append(float(target))
        if len(features) < self.min_samples:
            return None, None
        return np.vstack(features), np.asarray(labels, dtype=float)

    def _desired_state(self, *, symbol, ctx, current_price, current_value) -> DesiredState:
        del current_value
        closes = self._closes(ctx, symbol)
        if len(closes) < self.min_history:
            return None, "xgboost_insufficient_history"
        if float(current_price or 0.0) <= 0.0:
            return None, "xgboost_invalid_open_price"
        features, labels = self._dataset(closes)
        if features is None or labels is None:
            return None, "xgboost_insufficient_samples"
        latest_returns = closes.pct_change().dropna().tail(self.n_lags).to_numpy(dtype=float)
        if len(latest_returns) < self.n_lags:
            return None, "xgboost_insufficient_lags"
        try:
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=1,
                verbosity=0,
            )
            model.fit(features, labels)
            predicted_return = float(model.predict(latest_returns.reshape(1, -1))[0])
        except Exception as exc:
            return None, f"xgboost_fit_failed={type(exc).__name__}"
        return self._signal_from_predicted_return(
            predicted_return,
            "xgboost_return_forecast",
        )


PREDICTIVE_STRATEGIES: dict[str, Callable[[Dict], _BasePredictiveStrategy]] = {
    "arima": ARIMAStrategy,
    "xgboost": XGBoostStrategy,
}


def get_predictive_strategy_class(name: str) -> Callable[[Dict], _BasePredictiveStrategy] | None:
    return PREDICTIVE_STRATEGIES.get(str(name or "").strip().lower())
