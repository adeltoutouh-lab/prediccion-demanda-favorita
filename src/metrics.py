import numpy as np
import pandas as pd


def evaluate(y, pred):
    y, pred = np.asarray(y, dtype="float64"), np.asarray(pred, dtype="float64")
    if y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Valores no comparables para calcular métricas")
    error = pred - y
    total = np.abs(y).sum()
    return {"n": int(y.size), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "wape": float(np.abs(error).sum() / total) if total else None,
            "rmsle": float(np.sqrt(np.square(np.log1p(pred) - np.log1p(y)).mean())),
            "bias_pct": float(error.sum() / total) if total else None}


def grouped_metrics(predictions, keys):
    records = []
    for group, df in predictions.groupby(keys, observed=True, sort=True):
        if not isinstance(group, tuple):
            group = (group,)
        records.append(dict(zip(keys, group)) | evaluate(df["sales"], df["prediction"]))
    return pd.DataFrame(records)
