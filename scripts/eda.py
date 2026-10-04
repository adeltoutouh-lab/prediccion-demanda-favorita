"""Análisis descriptivo sobre todo el histórico, separado de la selección del modelo."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/eda"
FIGURES = ROOT / "docs/figures"


def save_figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                        "axes.spines.top": False, "axes.spines.right": False})
    train = pd.read_csv(ROOT / "data/raw/train.csv", parse_dates=["date"],
                        dtype={"store_nbr": "int16", "family": "category", "sales": "float32", "onpromotion": "int32"})
    train["is_promotion"] = train["onpromotion"].gt(0)
    monthly = train.groupby(train["date"].dt.to_period("M"), observed=True).agg(total_sales=("sales", "sum"), n=("sales", "size"))
    monthly["mean_sales"] = monthly["total_sales"] / monthly["n"]
    monthly.index = monthly.index.astype(str)
    monthly.to_csv(OUT / "monthly.csv")
    families = train.groupby("family", observed=True).agg(n=("sales", "size"),
        total_sales=("sales", "sum"), mean_sales=("sales", "mean"), median_sales=("sales", "median"),
        max_sales=("sales", "max"), zero_share=("sales", lambda x: x.eq(0).mean()),
        promotion_share=("is_promotion", "mean")).sort_values("total_sales", ascending=False)
    families.to_csv(OUT / "families.csv")
    promo = train.groupby(["family", "is_promotion"], observed=True)["sales"].agg(["mean", "size"])
    promo.to_csv(OUT / "promotion_by_family.csv")
    # Estas diferencias son descriptivas. No estiman el efecto causal de una promoción.
    recent = train.loc[train["date"].ge("2017-01-01")].copy()
    recent["weekday"] = recent["date"].dt.dayofweek
    weekday = recent.groupby("weekday")["sales"].mean().reindex(range(7))
    weekday.to_csv(OUT / "weekday_2017.csv")
    missing_dates = pd.date_range(train["date"].min(), train["date"].max()).difference(train["date"].unique())
    raw_quality = []
    for filename in ["train", "test", "stores", "transactions", "oil", "holidays_events"]:
        df = train.drop(columns="is_promotion") if filename == "train" else pd.read_csv(ROOT / f"data/raw/{filename}.csv")
        for column in df.columns:
            raw_quality.append({"file": filename, "column": column, "rows": len(df), "nulls": int(df[column].isna().sum())})
    pd.DataFrame(raw_quality).to_csv(OUT / "quality.csv", index=False)
    summary = {"rows": len(train), "stores": int(train["store_nbr"].nunique()), "families": int(train["family"].nunique()),
               "series": int(train[["store_nbr", "family"]].drop_duplicates().shape[0]),
               "start": str(train["date"].min().date()), "end": str(train["date"].max().date()),
               "observed_dates": int(train["date"].nunique()), "missing_dates": missing_dates.strftime("%Y-%m-%d").tolist(),
               "zero_share": float(train["sales"].eq(0).mean()), "negative_sales": int(train["sales"].lt(0).sum()),
               "duplicate_keys": int(train.duplicated(["date", "store_nbr", "family"]).sum()),
               "sales_quantiles": {str(q): float(train["sales"].quantile(q)) for q in [.5, .9, .99, 1.]},
               "promotion_share": float(train["is_promotion"].mean()),
               "top_family": str(families.index[0]),
               "top_family_volume_share": float(families["total_sales"].iloc[0] / families["total_sales"].sum()),
               "weekday_means_2017": weekday.to_dict()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(10, 3.9))
    dates = pd.to_datetime(monthly.index)
    ax.plot(dates, monthly["mean_sales"], color="#246d88", linewidth=2)
    ax.set(title="Ventas medias por registro y mes", ylabel="Ventas registradas por tienda y familia", xlabel="Mes")
    save_figure("01_monthly_sales")
    fig, ax = plt.subplots(figsize=(9.4, 4))
    ax.bar(["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"], weekday, color="#246d88")
    ax.set(title="Patrón semanal en 2017", ylabel="Ventas medias por registro", xlabel="Día de la semana")
    save_figure("02_weekday")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    top = families.head(10).iloc[::-1]
    ax.barh(top.index.astype(str), top["total_sales"] / 1_000_000, color="#246d88")
    ax.set(title="Familias con más ventas registradas", xlabel="Suma del campo sales (millones)")
    save_figure("03_family_volume")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    top_zero = families.sort_values("zero_share", ascending=False).head(10).iloc[::-1]
    ax.barh(top_zero.index.astype(str), top_zero["zero_share"] * 100, color="#ae7149")
    ax.set(title="Familias con más observaciones de ventas cero", xlabel="Registros con sales = 0 (%)")
    save_figure("04_zero_sales")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
