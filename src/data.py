"""Lectura, comprobaciones y construcción de las capas processed y gold."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

KEY = ["date", "store_nbr", "family"]
RAW_FILES = ["train.csv", "test.csv", "stores.csv", "transactions.csv",
             "holidays_events.csv", "oil.csv", "sample_submission.csv"]


def require_unique(df, keys, name):
    if df[keys].isna().any().any():
        raise ValueError(f"{name}: hay claves nulas")
    if df.duplicated(keys).any():
        raise ValueError(f"{name}: hay claves duplicadas {keys}")


def read_raw(folder):
    folder = Path(folder)
    missing = [f for f in RAW_FILES if not (folder / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos en {folder}: {missing}")
    raw = {}
    for filename in RAW_FILES:
        name = filename.removesuffix(".csv")
        df = pd.read_csv(folder / filename)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"], errors="raise")
        raw[name] = df
    for name in ["train", "test"]:
        df = raw[name]
        required = KEY + ["id", "onpromotion"] + (["sales"] if name == "train" else [])
        if not set(required).issubset(df):
            raise ValueError(f"{name}: columnas incorrectas")
        require_unique(df, KEY, name)
        require_unique(df, ["id"], name)
        if df[required].isna().any().any():
            raise ValueError(f"{name}: valores obligatorios nulos")
        if (df["onpromotion"] < 0).any():
            raise ValueError(f"{name}: promociones negativas")
        if name == "train" and ((df["sales"] < 0).any() or not np.isfinite(df["sales"]).all()):
            raise ValueError("train: ventas inválidas")
        df["store_nbr"] = df["store_nbr"].astype("int16")
        df["family"] = df["family"].astype("category")
        df["onpromotion"] = df["onpromotion"].astype("int32")
    raw["train"]["sales"] = raw["train"]["sales"].astype("float32")
    require_unique(raw["stores"], ["store_nbr"], "stores")
    require_unique(raw["transactions"], ["date", "store_nbr"], "transactions")
    require_unique(raw["oil"], ["date"], "oil")
    if (raw["transactions"]["transactions"] < 0).any():
        raise ValueError("transactions: valores negativos")
    if not set(raw["train"]["store_nbr"]).union(raw["test"]["store_nbr"]).issubset(raw["stores"]["store_nbr"]):
        raise ValueError("Hay tiendas sin metadatos")
    if not set(raw["test"]["family"]).issubset(raw["train"]["family"]):
        raise ValueError("El horizonte tiene familias sin histórico")
    if raw["test"]["date"].min() != raw["train"]["date"].max() + pd.Timedelta(days=1):
        raise ValueError("El horizonte no empieza después del histórico")
    return raw


def holidays_by_store(holidays, stores):
    """Expandir por ámbito y agregar ANTES del join con ventas."""
    rows = []
    for event in holidays.itertuples(index=False):
        if event.locale == "National":
            affected = stores
        elif event.locale == "Regional":
            affected = stores.loc[stores["state"].eq(event.locale_name)]
        elif event.locale == "Local":
            affected = stores.loc[stores["city"].eq(event.locale_name)]
        else:
            raise ValueError(f"Ámbito desconocido: {event.locale}")
        transferred = str(event.transferred).lower() == "true"
        effective = event.type in {"Holiday", "Additional", "Bridge", "Transfer"} and not transferred
        for store in affected["store_nbr"]:
            rows.append({"date": event.date, "store_nbr": store,
                         "is_holiday": int(effective), "is_event": int(event.type == "Event"),
                         "is_work_day": int(event.type == "Work Day"),
                         "is_transferred": int(transferred),
                         "holiday_type": event.type, "holiday_description": event.description})
    columns = ["date", "store_nbr", "is_holiday", "is_event", "is_work_day",
               "is_transferred", "holiday_type", "holiday_description"]
    if not rows:
        return pd.DataFrame(columns=columns)
    events = pd.DataFrame(rows)
    return events.groupby(["date", "store_nbr"], as_index=False).agg(
        is_holiday=("is_holiday", "max"), is_event=("is_event", "max"),
        is_work_day=("is_work_day", "max"), is_transferred=("is_transferred", "max"),
        holiday_type=("holiday_type", lambda x: "; ".join(sorted(set(x)))),
        holiday_description=("holiday_description", lambda x: "; ".join(sorted(set(x)))))


def build_gold(raw, root):
    root = Path(root)
    processed, gold_dir = root / "data/processed", root / "data/gold"
    processed.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    stores = raw["stores"].rename(columns={"type": "store_type"})
    holidays = holidays_by_store(raw["holidays_events"], raw["stores"])
    dates = pd.date_range(raw["train"]["date"].min(), raw["test"]["date"].max())
    oil = raw["oil"].set_index("date")["dcoilwtico"].reindex(dates)
    oil_daily = pd.DataFrame({"date": dates, "oil_price": oil.ffill().values,
                              "oil_price_imputed": oil.isna().values})
    stores.to_parquet(processed / "stores_clean.parquet", index=False)
    holidays.to_parquet(processed / "holidays_store_date.parquet", index=False)
    oil_daily.to_parquet(processed / "oil_daily.parquet", index=False)
    raw["transactions"].to_parquet(processed / "transactions_clean.parquet", index=False)

    output = {}
    for source, filename in [("train", "gold_sales_history"), ("test", "gold_forecast_horizon")]:
        df = raw[source].merge(stores, on="store_nbr", how="left", validate="many_to_one")
        df = df.merge(holidays, on=["date", "store_nbr"], how="left", validate="many_to_one")
        for col in ["is_holiday", "is_event", "is_work_day", "is_transferred"]:
            df[col] = df[col].fillna(0).astype("int8")
        for col in ["holiday_type", "holiday_description"]:
            df[col] = df[col].fillna("")
        if source == "train":
            df = df.merge(raw["transactions"], on=["date", "store_nbr"], how="left", validate="many_to_one")
            df["transactions_missing"] = df["transactions"].isna()
            # Los nulos se conservan: no sabemos si implican cierre o falta de registro.
        else:
            # El precio futuro realizado no estaría disponible en la fecha de corte.
            # Se conserva solo el último precio histórico conocido para el horizonte.
            last = oil_daily.loc[oil_daily["date"].le(raw["train"]["date"].max()), "oil_price"].dropna()
            df["oil_price"] = last.iloc[-1] if len(last) else np.nan
            df["oil_price_imputed"] = True
            df["oil_price_policy"] = "last_observed_at_cutoff"
        if source == "train":
            df = df.merge(oil_daily, on="date", how="left", validate="many_to_one")
        df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
        df["month"] = df["date"].dt.month.astype("int8")
        df["is_payday"] = (df["date"].dt.day.eq(15) | df["date"].dt.is_month_end).astype("int8")
        require_unique(df, KEY, filename)
        if len(df) != len(raw[source]):
            raise ValueError("Un join ha alterado el número de filas")
        df.to_parquet(gold_dir / f"{filename}.parquet", index=False)
        output[source] = df
    manifest = []
    for name in RAW_FILES:
        p = root / "data/raw" / name
        manifest.append({"file": name, "bytes": p.stat().st_size,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (root / "data/input_checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
