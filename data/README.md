# Datos y procedencia

Fuente original: [Kaggle · Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). El histórico tiene 3.000.888 filas, 54 tiendas y 33 familias. Termina el 15/08/2017; el horizonte sin etiquetas cubre el 16–31/08/2017.

## Obtención

```bash
python scripts/download_data.py
```

La ejecución entregada utilizó una **copia pública** del conjunto de la competición, alojada en un repositorio de terceros y fijada al commit `52b53a97fca77910f7ec1bf5cc58c4a334507c77`. La URL exacta, tamaños y hashes están en [source_manifest.json](source_manifest.json). El script verifica el SHA-256 del ZIP y extrae únicamente siete nombres permitidos. No se ha contrastado su hash con una descarga autenticada de Kaggle; sí se han comprobado columnas, claves, cobertura y periodo. Esto es una limitación de procedencia, no una certificación oficial del archivo.

Como alternativa, descarga el conjunto original desde Kaggle tras aceptar sus condiciones y descomprime estos archivos en `data/raw/`:

`train.csv`, `test.csv`, `stores.csv`, `transactions.csv`, `holidays_events.csv`, `oil.csv` y `sample_submission.csv`.

Los CSV originales no se publican aquí. El repositorio incluye resultados derivados y el extracto del histórico que utiliza el informe. La fuente y sus condiciones deben revisarse antes de reutilizar o redistribuir los datos fuera del proyecto académico.

## Capas

| Capa | Archivos | Clave y función |
|---|---|---|
| raw | Los siete CSV originales | Se conservan sin modificar |
| processed | `stores_clean`, `transactions_clean`, `holidays_store_date`, `oil_daily` en Parquet | Catálogos y contexto validados antes de los joins |
| gold | `gold_sales_history.parquet` | Fecha + tienda + familia; 3.000.888 filas con objetivo |
| gold | `gold_forecast_horizon.parquet` | Misma clave; 28.512 filas sin objetivo |

Los joins de contexto son `many_to_one`; el número de filas del histórico y del horizonte debe permanecer igual. Las transacciones y los festivos se relacionan por tienda y fecha; el catálogo de tiendas por tienda y el petróleo por fecha. Antes de unir los festivos se resuelve su ámbito nacional, regional o local y se agrupan las coincidencias para evitar duplicar ventas.

La capa gold no materializa las 27 variables de entrenamiento. Se calculan en `src/features.py` para cada corte temporal y cada estado recursivo, de modo que no puedan incluir información del horizonte. El calendario se completa allí; las cuatro fechas ausentes no se convierten en ventas cero.

El [diccionario de datos](diccionario.md) describe el resultado. [input_checksums.json](input_checksums.json) registra los archivos realmente leídos en la ejecución. Las capas locales quedan excluidas de Git porque se pueden regenerar.
