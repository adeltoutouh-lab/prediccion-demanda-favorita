"""Generar un proyecto Power BI completo (modelo TMSL y reporte PBIR).

Los CSV y los nombres de campo vienen del pipeline. El modelo usa archivos
públicos del propio repositorio por defecto, y permite cambiar a archivos locales.
"""
from pathlib import Path
import json
import uuid

ROOT = Path(__file__).resolve().parents[1]
PBI = ROOT / "powerbi"
REPORT = PBI / "Favorita.Report"
MODEL = PBI / "Favorita.SemanticModel"
SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/"


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


TABLES = {
    "Historico": ("historico.csv", {"date": "dateTime", "store_nbr": "int64", "family": "string", "sales": "double", "onpromotion": "int64", "is_holiday": "int64"}),
    "Predicciones": ("predicciones.csv", {"date": "dateTime", "store_nbr": "int64", "family": "string", "cutoff": "dateTime", "horizon_day": "int64", "model": "string", "predicted_sales": "double", "id": "int64", "onpromotion": "int64", "is_selected": "int64"}),
    "Validacion": ("validacion.csv", {"date": "dateTime", "store_nbr": "int64", "family": "string", "cutoff": "dateTime", "horizon_day": "int64", "model": "string", "role": "string", "prediction": "double", "sales": "double", "onpromotion": "int64", "absolute_error": "double", "squared_error": "double"}),
    "Tiendas": ("tiendas.csv", {"store_nbr": "int64", "city": "string", "state": "string", "store_type": "string", "cluster": "int64"}),
    "Familias": ("familias.csv", {"family": "string"}),
    "Modelos": ("modelos.csv", {"model": "string", "is_selected": "int64"}),
    "Calendario": ("calendario.csv", {"date": "dateTime", "year": "int64", "month": "int64"})
}


def m_source(filename, columns):
    m_types = {"string": "type text", "int64": "Int64.Type", "double": "type number", "dateTime": "type date"}
    types = ", ".join('{"' + name + '", ' + m_types[dtype] + '}' for name, dtype in columns.items())
    return ["let", f'    Archivo = if UseLocalFiles then File.Contents(LocalDataFolder & "/{filename}") else Web.Contents(DataBaseURL, [RelativePath="{filename}"]),',
            '    Fuente = Csv.Document(Archivo, [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
            "    Cabeceras = Table.PromoteHeaders(Fuente, [PromoteAllScalars=true]),",
            f'    Tipos = Table.TransformColumnTypes(Cabeceras, {{{types}}}, "en-US")', "in", "    Tipos"]


def build_model(selected):
    tables = []
    for name, (filename, columns) in TABLES.items():
        cols = []
        for column, dtype in columns.items():
            col = {"name": column, "dataType": dtype, "sourceColumn": column,
                   "summarizeBy": "none", "lineageTag": str(uuid.uuid5(uuid.NAMESPACE_URL, f"favorita/{name}/{column}"))}
            if dtype == "dateTime":
                col["formatString"] = "dd/MM/yyyy"
            if name == "Calendario" and column == "date":
                col["isKey"] = True
            cols.append(col)
        table = {"name": name, "columns": cols,
                 "partitions": [{"name": name, "mode": "import", "source": {"type": "m", "expression": m_source(filename, columns)}}]}
        if name == "Calendario":
            table["dataCategory"] = "Time"
        tables.append(table)
    # El modelo elegido es el valor por defecto si aún no se ha elegido un slicer.
    chosen = f'SELECTEDVALUE(Modelos[model], "{selected}")'
    select = f"VAR Modelo = {chosen} RETURN "
    def validation(expr):
        return select + f"CALCULATE({expr}, REMOVEFILTERS(Calendario), Modelos[model] = Modelo)"
    measures = {
        "Ventas historicas": ("SUM(Historico[sales])", "#,0.0"),
        "Ventas previstas": (select + "CALCULATE(SUM(Predicciones[predicted_sales]), Modelos[model] = Modelo)", "#,0.0"),
        "Promociones previstas": (select + "CALCULATE(SUM(Predicciones[onpromotion]), Modelos[model] = Modelo)", "#,0"),
        "Dia horizonte": ("MAX(Predicciones[horizon_day])", "0"),
        "MAE": (validation("AVERAGE(Validacion[absolute_error])"), "0.00"),
        "RMSE": (validation("SQRT(AVERAGE(Validacion[squared_error]))"), "0.00"),
        "WAPE": (validation("DIVIDE(SUM(Validacion[absolute_error]), SUM(Validacion[sales]))"), "0.0%"),
        "Sesgo": (validation("DIVIDE(SUM(Validacion[prediction]) - SUM(Validacion[sales]), SUM(Validacion[sales]))"), "0.0%"),
        "MAE baseline": ('CALCULATE([MAE], REMOVEFILTERS(Modelos), Modelos[model] = "baseline_semanal")', "0.00"),
        "Mejora MAE": ("DIVIDE([MAE baseline] - [MAE], [MAE baseline])", "0.0%"),
        "Ventas reales test": (validation("SUM(Validacion[sales])"), "#,0.0"),
        "Ventas previstas test": (validation("SUM(Validacion[prediction])"), "#,0.0"),
        "Modelo mostrado": (chosen, None),
        "Aviso historico": ('"Demostración histórica. Datos hasta agosto de 2017. Previsión del 16 al 31 de agosto de 2017."', None),
        "Aviso validacion": ('"Test interno independiente: 31/07/2017 a 15/08/2017. Las métricas conservan los filtros de tienda, familia y modelo."', None),
        "Aviso uso": ('"Ventas registradas previstas. Sin stock, precios ni costes, la previsión requiere revisión y no es una orden de compra."', None),
        "Estado datos": ('IF(ISBLANK([Ventas previstas]), "Sin predicciones para la selección", "Previsión disponible")', None)
    }
    # Las medidas viven en una tabla pequeña y oculta en la navegación de datos.
    tables.append({"name": "Medidas", "columns": [{"name": "dummy", "dataType": "int64", "sourceColumn": "dummy", "isHidden": True}],
                   "partitions": [{"name": "Medidas", "mode": "import", "source": {"type": "m", "expression": ["#table(type table [dummy = Int64.Type], {{0}})"]}}],
                   "measures": [{"name": name, "expression": expression, **({"formatString": fmt} if fmt else {})} for name, (expression, fmt) in measures.items()]})
    relationships = []
    for fact in ["Historico", "Predicciones", "Validacion"]:
        dims = [("Calendario", "date"), ("Tiendas", "store_nbr"), ("Familias", "family")]
        if fact != "Historico":
            dims.append(("Modelos", "model"))
        for dimension, key in dims:
            relationships.append({"name": str(uuid.uuid5(uuid.NAMESPACE_URL, f"favorita/{fact}/{dimension}")),
                                  "fromTable": fact, "fromColumn": key, "fromCardinality": "many",
                                  "toTable": dimension, "toColumn": key, "toCardinality": "one",
                                  "crossFilteringBehavior": "oneDirection", "isActive": True})
    expressions = [
        {"name": "DataBaseURL", "kind": "m", "expression": ['"https://raw.githubusercontent.com/adeltoutouh-lab/prediccion-demanda-favorita/main/powerbi/data/" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]']},
        {"name": "UseLocalFiles", "kind": "m", "expression": ['false meta [IsParameterQuery=true, Type="Logical", IsParameterQueryRequired=true]']},
        {"name": "LocalDataFolder", "kind": "m", "expression": ['"C:/TFM/prediccion-demanda-favorita/powerbi/data" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]']}
    ]
    write_json(MODEL / "model.bim", {"name": "Favorita", "compatibilityLevel": 1600,
        "model": {"culture": "es-ES", "sourceQueryCulture": "en-US", "defaultPowerBIDataSourceVersion": "powerBI_V3",
                  "dataAccessOptions": {"legacyRedirects": True, "returnErrorValuesAsNull": True},
                  "tables": tables, "relationships": relationships, "expressions": expressions,
                  "annotations": [{"name": "PBI_QueryOrder", "value": json.dumps(list(TABLES))},
                                  {"name": "__PBI_TimeIntelligenceEnabled", "value": "0"}]}})
    write_json(MODEL / "definition.pbism", {"$schema": SCHEMA + "item/semanticModel/definitionProperties/1.0.0/schema.json",
                                             "version": "1.0", "settings": {"qnaEnabled": False}})
    dax = "\n\n".join(f"// {name}\n{expression}" for name, (expression, _) in measures.items())
    (PBI / "medidas.dax").write_text(dax + "\n", encoding="utf-8")


def field(table, property, kind="Column"):
    return {kind: {"Expression": {"SourceRef": {"Entity": table}}, "Property": property}}


def projection(table, property, measure=False):
    return {"field": field(table, property, "Measure" if measure else "Column"),
            "queryRef": f"{table}.{property}", "nativeQueryRef": property}


def literal(value):
    return {"expr": {"Literal": {"Value": value}}}


def visual(page, name, kind, rect, roles=None, title=None, objects=None, sync=None):
    x, y, width, height = rect
    obj = {"$schema": SCHEMA + "item/report/definition/visualContainer/2.12.0/schema.json", "name": name,
           "position": {"x": x, "y": y, "z": 0, "width": width, "height": height, "tabOrder": 0},
           "visual": {"visualType": kind, "drillFilterOtherVisuals": True}}
    if roles:
        obj["visual"]["query"] = {"queryState": {role: {"projections": values} for role, values in roles.items()}}
    if title:
        obj["visual"]["visualContainerObjects"] = {"title": [{"properties": {"show": literal("true"), "text": literal("'" + title + "'"), "fontSize": literal("13D")}}]}
    if objects:
        obj["visual"]["objects"] = objects
    if sync:
        obj["visual"]["syncGroup"] = {"groupName": sync, "fieldChanges": True, "filterChanges": True}
    write_json(REPORT / "definition/pages" / page / "visuals" / name / "visual.json", obj)


def textbox(page, name, text, rect, size=18):
    visual(page, name, "textbox", rect, objects={"general": [{"properties": {"paragraphs": [
        {"textRuns": [{"value": text, "textStyle": {"fontFamily": "Segoe UI", "fontSize": f"{size}pt", "color": "#24445A"}}]}]}}]})


def slicers(page):
    for name, table, column, title, y in [("tienda", "Tiendas", "store_nbr", "Tienda", 140),
                                        ("familia", "Familias", "family", "Familia", 260),
                                        ("modelo", "Modelos", "model", "Modelo", 380)]:
        objects = {"data": [{"properties": {"mode": literal("'Dropdown'")}}]}
        if name == "modelo":
            objects["selection"] = [{"properties": {"singleSelect": literal("true")}}]
        visual(page, page + "_" + name, "slicer", (24, y, 215, 100),
               {"Values": [projection(table, column)]}, title, objects=objects, sync=name)


def build_report():
    definition = REPORT / "definition"
    write_json(PBI / "Favorita.pbip", {"$schema": SCHEMA + "pbip/pbipProperties/1.0.0/schema.json", "version": "1.0",
        "artifacts": [{"report": {"path": "Favorita.Report"}}], "settings": {"enableAutoRecovery": True}})
    write_json(REPORT / "definition.pbir", {"$schema": SCHEMA + "item/report/definitionProperties/2.0.0/schema.json",
        "version": "4.0", "datasetReference": {"byPath": {"path": "../Favorita.SemanticModel"}}})
    write_json(definition / "version.json", {"$schema": SCHEMA + "item/report/definition/versionMetadata/1.0.0/schema.json", "version": "2.0.0"})
    write_json(definition / "report.json", {"$schema": SCHEMA + "item/report/definition/report/3.3.0/schema.json",
        "themeCollection": {"baseTheme": {"name": "CY24SU06", "reportVersionAtImport": {"visual": "1.8.0", "page": "1.0.0", "report": "2.0.0"}, "type": "SharedResources"}},
        "settings": {"useStylableVisualContainerHeader": True, "exportDataMode": "AllowSummarized"}})
    write_json(definition / "pages/pages.json", {"$schema": SCHEMA + "item/report/definition/pagesMetadata/1.1.0/schema.json",
        "pageOrder": ["forecast", "evaluacion"], "activePageName": "forecast"})
    for name, label in [("forecast", "Previsión de ventas"), ("evaluacion", "Validación del modelo")]:
        write_json(definition / "pages" / name / "page.json", {"$schema": SCHEMA + "item/report/definition/page/2.1.0/schema.json",
            "name": name, "displayName": label, "displayOption": "FitToPage", "height": 800, "width": 1440})
        slicers(name)
    textbox("forecast", "forecast_titulo", "Corporación Favorita · Previsión de ventas", (24, 20, 1390, 55), 25)
    textbox("forecast", "forecast_aviso", "Demostración histórica: previsión del 16 al 31 de agosto de 2017", (24, 80, 1390, 38), 15)
    for i, (measure, label) in enumerate([("Ventas previstas", "Ventas previstas · 16 días"), ("MAE", "MAE · test interno"), ("WAPE", "WAPE · test interno"), ("Mejora MAE", "Mejora MAE vs baseline")]):
        visual("forecast", f"forecast_kpi_{i}", "card", (270 + i*285, 140, 270, 110),
               {"Values": [projection("Medidas", measure, True)]}, label)
    visual("forecast", "forecast_serie", "lineChart", (270, 280, 1135, 260),
           {"Category": [projection("Calendario", "date")], "Y": [projection("Medidas", "Ventas historicas", True), projection("Medidas", "Ventas previstas", True)]},
           "Histórico reciente y previsión de 16 días")
    visual("forecast", "forecast_detalle", "tableEx", (270, 570, 730, 175),
           {"Values": [projection("Predicciones", "date"), projection("Medidas", "Dia horizonte", True), projection("Medidas", "Ventas previstas", True), projection("Medidas", "Promociones previstas", True)]},
           "Detalle diario · menú (...) para exportar CSV")
    visual("forecast", "forecast_modelo_mostrado", "card", (24, 510, 215, 80),
           {"Values": [projection("Medidas", "Modelo mostrado", True)]}, "Modelo mostrado")
    visual("forecast", "forecast_estado", "card", (24, 610, 215, 95),
           {"Values": [projection("Medidas", "Estado datos", True)]}, "Estado de la previsión")
    visual("forecast", "forecast_horizonte", "lineChart", (1020, 570, 385, 175),
           {"Category": [projection("Validacion", "horizon_day")], "Y": [projection("Medidas", "MAE", True)]}, "MAE por día del horizonte")
    textbox("forecast", "forecast_limite", "La previsión necesita revisión. El dataset no contiene stock ni demanda insatisfecha.", (24, 755, 1380, 32), 12)

    textbox("evaluacion", "evaluacion_titulo", "Validación temporal · Resultados del test interno", (24, 20, 1390, 55), 25)
    textbox("evaluacion", "evaluacion_aviso", "31/07/2017 a 15/08/2017 · Modelo seleccionado antes de abrir este bloque", (24, 80, 1390, 38), 15)
    for i, (measure, label) in enumerate([("MAE", "MAE"), ("RMSE", "RMSE"), ("WAPE", "WAPE"), ("Sesgo", "Sesgo relativo")]):
        visual("evaluacion", f"evaluacion_kpi_{i}", "card", (270 + i*285, 140, 270, 110),
               {"Values": [projection("Medidas", measure, True)]}, label)
    visual("evaluacion", "evaluacion_modelos", "clusteredColumnChart", (270, 280, 535, 235),
           {"Category": [projection("Modelos", "model")], "Y": [projection("Medidas", "MAE", True)]}, "Comparación de modelos · MAE")
    visual("evaluacion", "evaluacion_horizonte", "lineChart", (840, 280, 565, 235),
           {"Category": [projection("Validacion", "horizon_day")], "Y": [projection("Medidas", "MAE", True)], "Series": [projection("Modelos", "model")]}, "Error a lo largo de los 16 días")
    visual("evaluacion", "evaluacion_familias", "tableEx", (270, 550, 1135, 195),
           {"Values": [projection("Familias", "family"), projection("Medidas", "MAE", True), projection("Medidas", "WAPE", True), projection("Medidas", "Mejora MAE", True)]}, "Error por familia · mismas fechas para todos los modelos")
    textbox("evaluacion", "evaluacion_limite", "WAPE pondera por volumen. Revise cada familia: una mejora global puede ocultar errores en familias pequeñas.", (24, 755, 1380, 32), 12)


if __name__ == "__main__":
    metadata = json.loads((ROOT / "outputs/run_metadata.json").read_text())
    name = metadata["selected_candidate"]
    selected = "lightgbm" if name.startswith("lightgbm") else name
    build_model(selected)
    build_report()
    print("Proyecto creado: powerbi/Favorita.pbip")
