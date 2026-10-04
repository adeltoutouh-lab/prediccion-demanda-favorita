from pathlib import Path
import json
import time
import platform
import importlib.metadata
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from .data import read_raw, build_gold
from .features import make_panel, training_data, recursive_forecast, FEATURES
from .models import build_model, fit_model
from .metrics import evaluate, grouped_metrics


def prediction_frame(panel, cutoff, values, name, role):
    rows = []
    start = panel.position(cutoff) + 1
    for step, prediction in enumerate(values, 1):
        df = panel.series[["store_nbr", "family"]].copy()
        df["date"] = panel.dates[start + step - 1]
        df["cutoff"] = pd.Timestamp(cutoff)
        df["horizon_day"] = step
        df["model"] = name
        df["role"] = role
        df["prediction"] = prediction
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def choose_model(validation_metrics, minimum_improvement):
    averaged = validation_metrics.groupby("model")["mae"].mean().sort_values()
    baseline = averaged["baseline_semanal"]
    for name in averaged.index:
        if name == "baseline_semanal":
            continue
        gains = []
        for cutoff, fold in validation_metrics.groupby("cutoff"):
            values = fold.set_index("model")["mae"]
            gains.append(1 - values[name] / values["baseline_semanal"])
        if 1 - averaged[name] / baseline >= minimum_improvement and min(gains) > 0:
            return name
    return "baseline_semanal"


def export_powerbi(root, gold, evaluation, final_forecasts, selected, lgb_name):
    folder = root / "powerbi/data"
    folder.mkdir(parents=True, exist_ok=True)
    names = {"baseline_semanal": "baseline_semanal", "ridge": "ridge", lgb_name: "lightgbm"}
    final_forecasts.rename(columns={"prediction": "predicted_sales"}).to_csv(folder / "predicciones.csv", index=False, float_format="%.6f")
    # El dashboard usa el test interno independiente, no el bloque usado para seleccionar.
    valid = evaluation.loc[evaluation["role"].eq("test_interno") & evaluation["model"].isin(names)].copy()
    valid["model"] = valid["model"].map(names)
    error = valid["prediction"].astype("float64") - valid["sales"].astype("float64")
    valid["absolute_error"] = error.abs()
    valid["squared_error"] = error ** 2
    valid.to_csv(folder / "validacion.csv", index=False, float_format="%.6f")
    history = gold["train"]
    recent = history.loc[history["date"].gt(history["date"].max() - pd.Timedelta(days=90)),
                         ["date", "store_nbr", "family", "sales", "onpromotion", "is_holiday"]]
    recent.to_csv(folder / "historico.csv", index=False, float_format="%.6f")
    stores = history[["store_nbr", "city", "state", "store_type", "cluster"]].drop_duplicates().sort_values("store_nbr")
    stores.to_csv(folder / "tiendas.csv", index=False)
    pd.DataFrame({"family": sorted(history["family"].unique())}).to_csv(folder / "familias.csv", index=False)
    pd.DataFrame({"model": list(names.values()), "is_selected": [int(n == selected) for n in names]}).to_csv(folder / "modelos.csv", index=False)
    dates = pd.DataFrame({"date": pd.date_range(recent["date"].min(), final_forecasts["date"].max())})
    dates["year"] = dates["date"].dt.year
    dates["month"] = dates["date"].dt.month
    dates.to_csv(folder / "calendario.csv", index=False)


def run(root, config):
    root = Path(root)
    started = time.perf_counter()
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    print("1. Lectura y capa gold", flush=True)
    raw = read_raw(root / "data/raw")
    gold = build_gold(raw, root)
    panel = make_panel(gold["train"], gold["test"])
    horizon = config["horizon"]
    if raw["test"]["date"].nunique() != horizon:
        raise ValueError("El horizonte de config no coincide con test.csv")
    expected_series = len(panel.series)
    for date, day in raw["test"].groupby("date"):
        if len(day) != expected_series:
            raise ValueError(f"Cobertura incompleta en {date}")
    cutoffs = config["validation_cutoffs"] + [config["holdout_cutoff"]]
    if sorted(cutoffs) != cutoffs or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Los cortes deben estar ordenados y ser únicos")
    for previous, following in zip(cutoffs, cutoffs[1:]):
        if (pd.Timestamp(following) - pd.Timestamp(previous)).days < horizon:
            raise ValueError("Los bloques de evaluación se solapan")
    candidates = ["baseline_semanal", "ridge"] + list(config["lightgbm_candidates"])
    results, fold_metrics, training_log = [], [], []
    actual = raw["train"][["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
    actual["family"] = actual["family"].astype(str)
    selected = None
    for cutoff in cutoffs:
        role = "test_interno" if cutoff == config["holdout_cutoff"] else "validacion"
        if role == "test_interno":
            # La elección queda fijada antes de leer las métricas del test interno.
            selected = choose_model(pd.DataFrame(fold_metrics), config["minimum_mae_improvement"])
            (out / "selection.json").write_text(json.dumps({"selected_candidate": selected,
                "selection_cutoffs": config["validation_cutoffs"],
                "minimum_mae_improvement": config["minimum_mae_improvement"],
                "rule": "menor MAE medio, mejora >=5% global y mejora en cada bloque de validación",
                "holdout_used_for_selection": False}, indent=2) + "\n")
            print(f"Modelo elegido ANTES del test interno: {selected}", flush=True)
        print(f"2. Corte {cutoff} ({role})", flush=True)
        x, y, train_info = training_data(panel, cutoff, config["train_window_days"])
        end = panel.position(cutoff)
        known_history = panel.sales[:end+1].copy()
        for name in candidates:
            clock = time.perf_counter()
            model = fit_model(build_model(name, config), name, x, y)
            trained = time.perf_counter()
            values = recursive_forecast(model, panel.context, known_history, horizon)
            generated = time.perf_counter()
            frame = prediction_frame(panel, cutoff, values, name, role)
            # Los valores reales se adjuntan después de generar todos los días.
            frame = frame.merge(actual, on=["date", "store_nbr", "family"], validate="one_to_one")
            if len(frame) != expected_series * horizon:
                raise ValueError("Faltan valores reales para el bloque completo")
            metric = {"cutoff": cutoff, "role": role, "model": name} | evaluate(frame["sales"], frame["prediction"])
            fold_metrics.append(metric)
            results.append(frame)
            training_log.append({"cutoff": cutoff, "model": name, **train_info,
                                 "fit_seconds": trained-clock, "forecast_seconds": generated-trained})
            print(f"  {name}: MAE={metric['mae']:.3f} WAPE={metric['wape']:.3%} tiempo={generated-clock:.1f}s", flush=True)
            pd.DataFrame(fold_metrics).to_csv(out / "metrics_by_fold.csv", index=False)
            (out / "training_log.json").write_text(json.dumps(training_log, indent=2) + "\n")
            del model
        del x, y
    evaluation = pd.concat(results, ignore_index=True)
    evaluation.to_parquet(out / "backtest_predictions.parquet", index=False)
    for label, keys in [("horizon", ["role", "model", "horizon_day"]),
                        ("family", ["role", "model", "family"]),
                        ("store", ["role", "model", "store_nbr"]),
                        ("promotion", ["role", "model", "is_promotion"])]:
        if label == "promotion":
            evaluation["is_promotion"] = evaluation["onpromotion"].gt(0)
        grouped_metrics(evaluation, keys).to_csv(out / f"metrics_by_{label}.csv", index=False)
    summary = grouped_metrics(evaluation, ["role", "model"])
    summary.to_csv(out / "metrics_summary.csv", index=False)
    lgb_name = summary.loc[summary["role"].eq("validacion") & summary["model"].str.startswith("lightgbm")].sort_values("mae")["model"].iloc[0]
    if selected.startswith("lightgbm"):
        lgb_name = selected
    print("3. Entrenamiento final y forecast 2017-08-16 a 2017-08-31", flush=True)
    final_cutoff = str(raw["train"]["date"].max().date())
    x, y, train_info = training_data(panel, final_cutoff, config["train_window_days"])
    end = panel.position(final_cutoff)
    forecasts = []
    names = {"baseline_semanal": "baseline_semanal", "ridge": "ridge", lgb_name: "lightgbm"}
    for name, public_name in names.items():
        clock = time.perf_counter()
        model = fit_model(build_model(name, config), name, x, y)
        trained = time.perf_counter()
        values = recursive_forecast(model, panel.context, panel.sales[:end+1].copy(), horizon)
        frame = prediction_frame(panel, final_cutoff, values, public_name, "forecast")
        frame = frame.drop(columns="role").merge(panel.future_ids, on=["date", "store_nbr", "family"], validate="one_to_one")
        frame["is_selected"] = int(name == selected)
        forecasts.append(frame)
        training_log.append({"cutoff": final_cutoff, "model": name, **train_info,
                             "fit_seconds": trained-clock, "forecast_seconds": time.perf_counter()-trained})
        if name.startswith("lightgbm"):
            model.booster_.save_model(str(root / "models/lightgbm.txt"))
            importance = pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain"),
                                       "split_count": model.booster_.feature_importance("split")}).sort_values("gain", ascending=False)
            importance.to_csv(out / "feature_importance.csv", index=False)
        elif name == "ridge":
            joblib.dump(model, root / "models/ridge.joblib")
        del model
    final_forecasts = pd.concat(forecasts, ignore_index=True)
    final_forecasts.to_parquet(out / "forecast.parquet", index=False)
    submission = final_forecasts.loc[final_forecasts["is_selected"].eq(1), ["id", "prediction"]].rename(columns={"prediction": "sales"})
    submission = raw["sample_submission"][["id"]].merge(submission, on="id", how="left", validate="one_to_one")
    if submission["sales"].isna().any() or len(submission) != len(raw["test"]):
        raise ValueError("Submission incompleta")
    submission.to_csv(out / "submission.csv", index=False, float_format="%.6f")
    export_powerbi(root, gold, evaluation, final_forecasts, selected, lgb_name)
    (out / "training_log.json").write_text(json.dumps(training_log, indent=2) + "\n")
    (out / "run_metadata.json").write_text(json.dumps({
        "executed_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "lightgbm", "pyarrow", "scipy"]},
        "config": config, "features": FEATURES, "selected_candidate": selected,
        "lightgbm_dashboard_candidate": lgb_name, "raw_rows": len(raw["train"]),
        "series": expected_series, "horizon": horizon, "submission_rows": len(submission),
        "elapsed_seconds": time.perf_counter()-started,
        "forecast_semantics": "ventas históricas previstas, no demanda ni previsión actual",
        "future_transactions_used": False, "future_oil_used": False}, indent=2) + "\n")
    print(f"Terminado. {len(submission):,} predicciones. {time.perf_counter()-started:.1f}s", flush=True)
