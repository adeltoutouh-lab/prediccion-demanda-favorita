"""Variables con retardos de calendario y estado separado para cada modelo."""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd

CATEGORICAL = ["store_nbr", "family_code", "series_code", "city_code", "state_code", "type_code", "cluster", "day_of_week", "month"]
NUMERIC = ["day_index", "year_sin", "year_cos", "is_payday", "is_holiday", "is_event", "is_work_day", "promotion_log", "is_promotion",
           "lag_1", "lag_7", "lag_14", "lag_28", "mean_7", "mean_28", "std_28", "weekday_mean_4", "missing_lags"]
FEATURES = CATEGORICAL + NUMERIC


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    series: pd.DataFrame
    sales: np.ndarray
    context: np.ndarray
    future_ids: pd.DataFrame

    def position(self, date):
        return int(self.dates.get_loc(pd.Timestamp(date)))


def make_panel(history, future):
    """La cuadrícula mantiene NaN en fechas sin observación; no inventa ventas cero."""
    series = history[["store_nbr", "family"]].drop_duplicates().sort_values(["store_nbr", "family"]).reset_index(drop=True)
    series["family"] = series["family"].astype(str)
    series["series_code"] = np.arange(len(series), dtype="int32")
    dates = pd.date_range(history["date"].min(), future["date"].max())
    both = pd.concat([history, future], ignore_index=True)
    both["family"] = both["family"].astype(str)
    both = both.merge(series, on=["store_nbr", "family"], validate="many_to_one")
    date_pos = ((both["date"] - dates[0]).dt.days).to_numpy()
    series_pos = both["series_code"].to_numpy()
    sales = np.full((len(dates), len(series)), np.nan, dtype="float32")
    historical = both["sales"].notna().to_numpy()
    sales[date_pos[historical], series_pos[historical]] = both.loc[historical, "sales"].to_numpy()

    metadata = both.sort_values("date").drop_duplicates("series_code").set_index("series_code").loc[series["series_code"]]
    context = np.zeros((len(dates), len(series), len(CATEGORICAL) + 9), dtype="float32")
    for i, name in enumerate(["store_nbr", "family", "series_code", "city", "state", "store_type", "cluster"]):
        values = series["series_code"] if name == "series_code" else metadata[name]
        values = values.to_numpy() if name in {"store_nbr", "series_code", "cluster"} else pd.Categorical(values, categories=sorted(values.unique())).codes
        context[:, :, i] = values
    context[:, :, 7] = dates.dayofweek.to_numpy()[:, None]
    context[:, :, 8] = dates.month.to_numpy()[:, None]
    context[:, :, 9] = np.arange(len(dates))[:, None] / 365.25
    context[:, :, 10] = np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)[:, None]
    context[:, :, 11] = np.cos(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)[:, None]
    context[:, :, 12] = ((dates.day == 15) | dates.is_month_end)[:, None]
    for i, col in enumerate(["is_holiday", "is_event", "is_work_day"], 13):
        context[date_pos, series_pos, i] = both[col].to_numpy()
    context[date_pos, series_pos, 16] = np.log1p(both["onpromotion"].to_numpy())
    context[date_pos, series_pos, 17] = both["onpromotion"].gt(0).to_numpy()
    return Panel(dates, series, sales, context, future[["id", "date", "store_nbr", "family", "onpromotion"]].copy())


def features_for_day(context, sales, day):
    if day < 28:
        raise ValueError("Se necesitan 28 días previos")
    # Todo lo leído de sales tiene índice estrictamente menor que day.
    lags = np.stack([sales[day - lag] for lag in [1, 7, 14, 28]], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.stack([np.nanmean(sales[day-7:day], axis=0),
                          np.nanmean(sales[day-28:day], axis=0),
                          np.nanstd(sales[day-28:day], axis=0),
                          np.nanmean(sales[[day-7, day-14, day-21, day-28]], axis=0)], axis=1)
    missing = np.isnan(lags).sum(axis=1, keepdims=True)
    dynamic = np.concatenate([lags, means, missing], axis=1)
    x = np.concatenate([context[day], dynamic], axis=1)
    return np.nan_to_num(x, nan=0.0).astype("float32")


def training_data(panel, cutoff, window_days):
    end = panel.position(cutoff)
    start = max(28, end - window_days + 1)
    x = np.concatenate([features_for_day(panel.context, panel.sales, t) for t in range(start, end + 1)])
    y = panel.sales[start:end+1].reshape(-1)
    mask = np.isfinite(y)
    return x[mask], y[mask], {"train_start": str(panel.dates[start].date()),
                             "train_end": str(panel.dates[end].date()), "train_rows": int(mask.sum())}


def recursive_forecast(model, context, history, horizon):
    """Recibe exclusivamente el histórico hasta el corte y contexto sin ventas."""
    start = len(history)
    if start + horizon > len(context):
        raise ValueError("No hay contexto para todo el horizonte")
    state = np.full((start + horizon, history.shape[1]), np.nan, dtype="float32")
    state[:start] = history
    predictions = []
    for day in range(start, start + horizon):
        if model is None:
            pred = state[day-7].copy()
            # Respaldo cuando falta la fecha t-7, manteniendo la restricción temporal.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fallback = np.nanmean(state[day-28:day], axis=0)
            pred = np.where(np.isfinite(pred), pred, np.nan_to_num(fallback, nan=0.0))
        else:
            x = features_for_day(context, state, day)
            if hasattr(model, "feature_name_"):
                x = pd.DataFrame(x, columns=model.feature_name_)
            pred = model.predict(x)
        pred = np.maximum(0, np.asarray(pred, dtype="float32"))
        if not np.isfinite(pred).all():
            raise ValueError("El modelo produjo una predicción no finita")
        state[day] = pred
        predictions.append(pred)
    return np.stack(predictions)
