"""Pruebas de reglas que pueden invalidar el resultado, no del ajuste del modelo."""
import numpy as np
import pandas as pd
import pytest
from src.features import features_for_day, recursive_forecast, FEATURES
from src.metrics import evaluate
from src.data import holidays_by_store, require_unique
from src.pipeline import choose_model


def test_baseline_reuses_predictions_after_day_seven():
    history = np.arange(1, 36, dtype="float32")[:, None]
    context = np.zeros((51, 1, 18), dtype="float32")
    predicted = recursive_forecast(None, context, history, 16).ravel()
    np.testing.assert_array_equal(predicted[:7], np.arange(29, 36))
    np.testing.assert_array_equal(predicted[7:14], predicted[:7])
    np.testing.assert_array_equal(predicted[14:], predicted[:2])


def test_features_ignore_current_and_future_sales():
    sales = np.arange(60, dtype="float32")[:, None]
    context = np.zeros((60, 1, 18), dtype="float32")
    before = features_for_day(context, sales, 35)
    sales[35:] = 999_999
    np.testing.assert_array_equal(before, features_for_day(context, sales, 35))


def test_missing_calendar_date_does_not_shift_the_lag():
    sales = np.arange(60, dtype="float32")[:, None]
    sales[31] = np.nan
    context = np.zeros((60, 1, 18), dtype="float32")
    x = features_for_day(context, sales, 40)
    assert x[0, FEATURES.index("lag_7")] == 33
    assert x[0, FEATURES.index("lag_14")] == 26


def test_each_model_has_its_own_recursive_history():
    class Constant:
        def predict(self, x):
            return np.full(len(x), 100)
    history = np.arange(35, dtype="float32")[:, None]
    original = history.copy()
    context = np.zeros((51, 1, 18), dtype="float32")
    recursive_forecast(Constant(), context, history, 16)
    np.testing.assert_array_equal(history, original)
    np.testing.assert_array_equal(recursive_forecast(None, context, history, 16).ravel()[:7], np.arange(28, 35))


def test_metrics_weight_volume_and_handle_zero_denominator():
    metrics = evaluate([0, 100], [10, 90])
    assert metrics["mae"] == 10
    assert metrics["wape"] == .2
    assert evaluate([0, 0], [1, 0])["wape"] is None
    with pytest.raises(ValueError):
        evaluate([1, np.nan], [1, 1])


def test_holidays_respect_locality_and_transfers_without_duplicating_sales():
    stores = pd.DataFrame({"store_nbr": [1, 2], "city": ["Quito", "Guayaquil"], "state": ["Pichincha", "Guayas"]})
    holidays = pd.DataFrame([
        ["2017-01-01", "Holiday", "Local", "Quito", "Local", False],
        ["2017-01-01", "Event", "National", "Ecuador", "Evento", False],
        ["2017-01-02", "Holiday", "National", "Ecuador", "Original trasladado", True],
        ["2017-01-03", "Transfer", "National", "Ecuador", "Día efectivo", False],
    ], columns=["date", "type", "locale", "locale_name", "description", "transferred"])
    holidays["date"] = pd.to_datetime(holidays["date"])
    expanded = holidays_by_store(holidays, stores)
    require_unique(expanded, ["date", "store_nbr"], "festivos")
    values = expanded.set_index(["date", "store_nbr"])
    assert values.loc[(pd.Timestamp("2017-01-01"), 1), "is_holiday"] == 1
    assert values.loc[(pd.Timestamp("2017-01-01"), 2), "is_holiday"] == 0
    assert values.loc[(pd.Timestamp("2017-01-02"), 1), "is_holiday"] == 0
    assert values.loc[(pd.Timestamp("2017-01-03"), 1), "is_holiday"] == 1


def test_duplicates_are_rejected_instead_of_multiplying_rows():
    with pytest.raises(ValueError, match="duplicadas"):
        require_unique(pd.DataFrame({"key": [1, 1]}), ["key"], "datos")


def test_selection_requires_improvement_in_all_validation_blocks():
    metrics = pd.DataFrame([
        ["A", "baseline_semanal", 100], ["B", "baseline_semanal", 100],
        ["A", "ridge", 10], ["B", "ridge", 110],
        ["A", "lightgbm_31", 80], ["B", "lightgbm_31", 85],
    ], columns=["cutoff", "model", "mae"])
    assert choose_model(metrics, .05) == "lightgbm_31"
