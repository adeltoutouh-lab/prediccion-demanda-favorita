"""Comprobar esquemas PBIR, referencias y consistencia de los datos.

Uso: python scripts/validate_powerbi.py --schemas /ruta/a/microsoft-json-schemas
Los esquemas oficiales están en https://github.com/microsoft/json-schemas.
Esta comprobación no sustituye abrir y refrescar el proyecto en Power BI Desktop.
"""
from pathlib import Path
import argparse
import json
import warnings
import pandas as pd
import numpy as np
from urllib.parse import urlparse, unquote
import jsonschema
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def validate(schema_root=None):
    pbi = ROOT / "powerbi"
    model = json.loads((pbi / "Favorita.SemanticModel/model.bim").read_text())["model"]
    tables = {t["name"]: t for t in model["tables"]}
    members = {name: {v["name"] for group in ["columns", "measures"] for v in table.get(group, [])} for name, table in tables.items()}
    checks = 0
    for path in list(pbi.rglob("*.json")) + list(pbi.rglob("*.pbip")) + list(pbi.rglob("*.pbir")) + list(pbi.rglob("*.pbism")):
        doc = json.loads(path.read_text())
        if "$schema" in doc and schema_root:
            def load_schema(url):
                if url.startswith("file:"):
                    target = Path(unquote(urlparse(url).path))
                else:
                    relative = urlparse(url).path.split("/json-schemas/", 1)[-1]
                    target = Path(schema_root) / relative
                if not target.exists() and target.name == "schema.embedded.json":
                    target = target.with_name("schema-embedded.json")
                if target.exists():
                    return json.loads(target.read_text())
                # Algunas versiones publican esquemas embedded generados que no
                # forman parte del árbol fuente. Se resuelven desde Microsoft.
                with urllib.request.urlopen(url, timeout=30) as response:
                    return json.load(response)
            schema = load_schema(doc["$schema"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                resolver = jsonschema.RefResolver(base_uri=doc["$schema"], referrer=schema,
                                                 handlers={"https": load_schema, "file": load_schema})
                jsonschema.Draft7Validator(schema, resolver=resolver).validate(doc)
        if path.name == "visual.json":
            query = doc.get("visual", {}).get("query", {}).get("queryState", {})
            for role in query.values():
                for item in role["projections"]:
                    body = next(iter(item["field"].values()))
                    table = body["Expression"]["SourceRef"]["Entity"]
                    column = body["Property"]
                    if table not in members or column not in members[table]:
                        raise ValueError(f"Referencia inexistente: {table}.{column}")
                    checks += 1
    for relationship in model["relationships"]:
        for side in ["from", "to"]:
            assert relationship[side + "Column"] in members[relationship[side + "Table"]]
    datasets = {}
    for table in tables.values():
        if table["name"] == "Medidas":
            continue
        source = "\n".join(table["partitions"][0]["source"]["expression"])
        import re
        filename = re.search(r'RelativePath="([^"]+)"', source)[1]
        df = pd.read_csv(pbi / "data" / filename)
        datasets[table["name"]] = df
        assert set(c["sourceColumn"] for c in table["columns"]).issubset(df)
        assert not df[[c["sourceColumn"] for c in table["columns"]]].isna().any().any()
    for relation in model["relationships"]:
        one = datasets[relation["toTable"]][relation["toColumn"]]
        many = datasets[relation["fromTable"]][relation["fromColumn"]]
        assert one.is_unique
        assert many.isin(one).all()
    forecast = datasets["Predicciones"]
    valid = datasets["Validacion"]
    assert not forecast.duplicated(["date", "store_nbr", "family", "model"]).any()
    assert not valid.duplicated(["date", "store_nbr", "family", "model"]).any()
    assert forecast.groupby("model").size().eq(28_512).all()
    assert set(valid["model"]) == set(datasets["Modelos"]["model"])
    assert valid.groupby("model").size().eq(28_512).all()
    assert (forecast["predicted_sales"] >= 0).all()
    assert ((valid["prediction"] - valid["sales"]).abs() - valid["absolute_error"]).abs().max() < 2e-6
    metadata = json.loads((ROOT / "outputs/run_metadata.json").read_text())
    metrics = pd.read_csv(ROOT / "outputs/metrics_summary.csv")
    expected = metrics.loc[metrics["role"].eq("test_interno")].set_index("model")
    aliases = {"baseline_semanal": "baseline_semanal", "ridge": "ridge",
               "lightgbm": metadata["lightgbm_dashboard_candidate"]}
    for name, group in valid.groupby("model"):
        error = group["prediction"] - group["sales"]
        actual = [error.abs().mean(), np.sqrt((error ** 2).mean()), error.abs().sum() / group["sales"].sum()]
        np.testing.assert_allclose(actual, expected.loc[aliases[name], ["mae", "rmse", "wape"]].to_numpy(dtype=float), rtol=1e-7)
    result = {"schema_validation": bool(schema_root), "field_references": checks,
              "tables": len(tables), "relationships": len(model["relationships"]),
              "visuals": len(list(pbi.rglob("visual.json"))), "metric_reconciliation": True,
              "validation_rows_per_model": 28512, "desktop_opened": False}
    (ROOT / "outputs/powerbi_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemas")
    validate(parser.parse_args().schemas)
