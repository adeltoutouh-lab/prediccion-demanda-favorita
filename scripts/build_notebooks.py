"""Crear y ejecutar los notebooks a partir del análisis y resultados del pipeline."""
from pathlib import Path
import argparse
import os
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def execute_in_process(notebook):
    """Ejecutar con IPython cuando el entorno no permite sockets de Jupyter."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    os.chdir(ROOT)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        result.raise_error()
        cell.execution_count = count
        cell.outputs = []
        for name in ("stdout", "stderr"):
            value = getattr(captured, name)
            if value:
                cell.outputs.append(nbf.v4.new_output("stream", name=name, text=value))
        for output in captured.outputs:
            cell.outputs.append(nbf.v4.new_output("display_data", data=output.data, metadata=output.metadata))
    notebook.metadata["execution"] = {"runner": "IPython in-process", "reason": "Entorno sin sockets de kernel"}


def build(filename, cells):
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.cells = [nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text) for kind, text in cells]
    if ARGS.in_process:
        execute_in_process(notebook)
    else:
        NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(notebook, OUT / filename)
    print("Ejecutado:", filename)


setup = '''from pathlib import Path
import json
import pandas as pd
import numpy as np
from IPython.display import display, Image
root = Path.cwd()
if not (root / "outputs").exists():
    root = root.parent
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-process", action="store_true", help="Ejecutar con IPython sin abrir un kernel por sockets")
    ARGS = parser.parse_args()
    build("01_datos_y_eda.ipynb", [
        ("md", """# Datos y análisis exploratorio\n\nTFM de predicción de ventas de Corporación Favorita. Trabajo con una fila por fecha, tienda y familia. Las figuras y tablas siguientes salen del histórico real utilizado en la ejecución, no de datos de ejemplo.\n\nEl cálculo reproducible está en `scripts/eda.py`. Para regenerar las tablas: `python scripts/eda.py`. Primero hay que descargar los CSV con `scripts/download_data.py`."""),
        ("code", setup),
        ("code", '''summary = json.loads((root / "outputs/eda/summary.json").read_text())
display(pd.Series({k:summary[k] for k in ["rows", "stores", "families", "series", "start", "end", "observed_dates", "zero_share", "duplicate_keys", "negative_sales"]}))
print("Fechas sin observaciones:", summary["missing_dates"])'''),
        ("md", """## Calidad del dato\n\nLa ausencia de duplicados no basta para modelar. Faltan cuatro fechas completas, todas el 25 de diciembre. Para los lags he mantenido el calendario completo y ventas desconocidas en esas fechas. No las he convertido en ventas cero.\n\nLas transacciones y el precio del petróleo no entran como variables del mismo día del forecast. No estarían disponibles al generar una previsión."""),
        ("code", '''quality = pd.read_csv(root / "outputs/eda/quality.csv")
display(quality.loc[quality["nulls"].gt(0)])
display(pd.read_json(root / "data/input_checksums.json")[["file", "bytes"]])'''),
        ("md", """## Patrón temporal\n\nEl patrón semanal justifica comparar el modelo con la venta de la semana anterior. La serie mensual muestra la evolución descriptiva del campo sales. No utilizo el EDA del histórico completo para seleccionar hiperparámetros con el test interno."""),
        ("code", '''display(Image(filename=str(root / "docs/figures/01_monthly_sales.png")))
display(Image(filename=str(root / "docs/figures/02_weekday.png")))'''),
        ("md", """## Familias y ceros\n\nLas familias tienen escalas diferentes y muchas observaciones cero. Un buen MAE global puede ocultar problemas de algunas familias. Tampoco puedo saber si un cero corresponde a poca venta, un cierre o una rotura de stock."""),
        ("code", '''families = pd.read_csv(root / "outputs/eda/families.csv")
display(families.head(10).round(3))
display(Image(filename=str(root / "docs/figures/04_zero_sales.png")))'''),
        ("md", """## Promociones\n\nLa siguiente tabla compara medias y tamaños de grupo por familia. Es una asociación descriptiva. Las promociones pueden coincidir con productos o fechas de mayor venta y esta comparación no demuestra causalidad."""),
        ("code", '''promotion = pd.read_csv(root / "outputs/eda/promotion_by_family.csv")
display(promotion.loc[promotion["family"].isin(["GROCERY I", "PRODUCE", "DAIRY"])].round(2))'''),
        ("md", """## Decisiones para el modelado\n\nConservo los ceros y los valores altos. Uso calendario, metadatos, promociones y ventas anteriores. La evaluación se hace en bloques completos de 16 días. El forecast debe reutilizar sus propias predicciones cuando un retardo entra en ese bloque. Las comprobaciones de calidad y los joins están implementados en `src/data.py`.""")
    ])
    build("02_modelado_y_resultados.ipynb", [
        ("md", """# Modelado y evaluación temporal\n\nLa ejecución completa está en `run_pipeline.py` y los módulos `src/`. Este notebook revisa sus salidas y recalcula las métricas del dashboard para comprobar que coinciden con la evaluación. Para entrenar de nuevo: `python run_pipeline.py`.\n\nEl modelo se elige con tres bloques de validación. El último bloque, 31/07/2017 a 15/08/2017, es un test interno independiente. El test oficial de Kaggle no tiene etiquetas y no produce una puntuación local."""),
        ("code", setup),
        ("code", '''metadata = json.loads((root / "outputs/run_metadata.json").read_text())
selection = json.loads((root / "outputs/selection.json").read_text())
display(selection)
display(metadata["config"])
print("Variables:", metadata["features"])'''),
        ("md", """## Comparación en los tres bloques de validación\n\nBaseline y modelos predicen recursivamente bajo la misma restricción. El baseline no puede consultar ventas reales del horizonte a partir del día 8. Ridge sirve como alternativa lineal y las dos configuraciones de LightGBM permiten una comparación pequeña y reproducible."""),
        ("code", '''folds = pd.read_csv(root / "outputs/metrics_by_fold.csv")
display(folds.loc[folds["role"].eq("validacion")].pivot(index="cutoff", columns="model", values="mae").round(3))
assert selection["holdout_used_for_selection"] is False'''),
        ("md", """## Resultado del test interno\n\nEstas métricas no han intervenido en la selección. MAE se expresa en la escala de sales. WAPE es la suma del error absoluto dividida entre las ventas totales. RMSLE complementa la comparación y ayuda a ver diferencias en la escala logarítmica."""),
        ("code", '''metrics = pd.read_csv(root / "outputs/metrics_summary.csv")
test = metrics.loc[metrics["role"].eq("test_interno")].set_index("model")
display(test[["n", "mae", "rmse", "wape", "rmsle", "bias_pct"]].round(4))
winner = metadata["selected_candidate"]
improvement = 1 - test.loc[winner, "mae"] / test.loc["baseline_semanal", "mae"]
print(f"Mejora MAE del modelo elegido: {improvement:.2%}")'''),
        ("md", """## Comprobación de las métricas desde los CSV de Power BI\n\nEl CSV conserva el detalle de errores en precisión de 64 bits antes de redondearlo para exportar. Recalculo las métricas con las mismas filas, sin promediar porcentajes por tienda o familia."""),
        ("code", '''valid = pd.read_csv(root / "powerbi/data/validacion.csv")
rows = []
alias = {"baseline_semanal":"baseline_semanal", "ridge":"ridge", "lightgbm":metadata["lightgbm_dashboard_candidate"]}
for model, group in valid.groupby("model"):
    error = group["prediction"] - group["sales"]
    mae = error.abs().mean()
    rmse = np.sqrt((error**2).mean())
    wape = error.abs().sum() / group["sales"].sum()
    expected = test.loc[alias[model]]
    assert np.isclose(mae, expected["mae"], rtol=1e-7)
    assert np.isclose(rmse, expected["rmse"], rtol=1e-7)
    assert np.isclose(wape, expected["wape"], rtol=1e-7)
    rows.append({"model":model, "mae":mae, "rmse":rmse, "wape":wape})
display(pd.DataFrame(rows).round(5))'''),
        ("md", """## Comprobación del baseline a partir del octavo día\n\nLas predicciones de los días 8-14 repiten las del 1-7. Esto es lo que corresponde a un baseline semanal recursivo, sin abrir las etiquetas reales del bloque."""),
        ("code", '''baseline = valid.loc[valid["model"].eq("baseline_semanal")]
pivot = baseline.pivot(index=["store_nbr","family"], columns="horizon_day", values="prediction")
np.testing.assert_allclose(pivot[list(range(8,15))].to_numpy(), pivot[list(range(1,8))].to_numpy())
print("Baseline recursivo comprobado en las 1.782 series.")'''),
        ("md", """## Error por horizonte y familia\n\nEn este test el error no crece de forma uniforme con el horizonte. Hay picos en fechas concretas. La recursión puede acumular error, pero no se puede atribuir toda la curva únicamente a ese efecto."""),
        ("code", '''display(Image(filename=str(root / "docs/figures/06_horizon_error.png")))
segments = pd.read_csv(root / "outputs/metrics_by_family.csv")
display(segments.loc[segments["role"].eq("test_interno") & segments["model"].eq(winner)].sort_values("mae",ascending=False).head(8).round(3))'''),
        ("md", """## Forecast final y límites\n\nLas 28.512 predicciones finales corresponden al 16-31 de agosto de 2017. No hay ventas reales de ese bloque para evaluarlas aquí. Las métricas de calidad mostradas en Power BI corresponden al test interno, con sus fechas visibles. No calculo demanda insatisfecha ni ahorro económico porque no hay stock, precios o costes."""),
        ("code", '''forecast = pd.read_csv(root / "powerbi/data/predicciones.csv")
submission = pd.read_csv(root / "outputs/submission.csv")
assert forecast.groupby("model").size().eq(28512).all()
assert len(submission) == 28512 and submission["id"].is_unique
assert submission["sales"].ge(0).all() and np.isfinite(submission["sales"]).all()
display(submission.head())
display(Image(filename=str(root / "docs/figures/07_forecast_example.png")))''')
    ])
