# Predicción de ventas en Corporación Favorita

Proyecto académico de Data Science / IA desarrollado por **Adel Toutouh El Bouchti**.

## Objetivo

Desarrollar un sistema de predicción de **ventas diarias** por tienda y familia de producto utilizando el dataset público **Store Sales - Time Series Forecasting** de Kaggle.

El proyecto utiliza `sales` como variable objetivo. Por tanto, el alcance se limita a predecir ventas registradas y no demanda real ni decisiones automáticas de inventario.

## Estado actual

El repositorio contiene las cuatro primeras entregas del proyecto:

- `docs/entregas/01_ideas_producto.md`
- `docs/entregas/02_datos_necesarios.md`
- `docs/entregas/03_modelo_datos.md`
- `docs/entregas/04_analisis_modelado.md`

## Estructura

```text
.
|-- README.md
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- gold/
|-- docs/
|   `-- entregas/
|       |-- 01_ideas_producto.md
|       |-- 02_datos_necesarios.md
|       |-- 03_modelo_datos.md
|       `-- 04_analisis_modelado.md
|-- notebooks/
|-- reports/
`-- src/
```

## Fuente de datos

Los datos proceden de la competición **Store Sales - Time Series Forecasting** de Kaggle.

Es necesario disponer de una cuenta gratuita de Kaggle y aceptar las condiciones de la competición para descargar los archivos.

Los archivos originales y los datasets generados no se incluyen en el repositorio. Solo se publican código, documentación, resultados agregados y archivos README de las capas de datos.

## Alcance del MVP

El MVP previsto incluye:

- pipeline reproducible en Python;
- baseline sencillo;
- al menos un modelo mejorado;
- validación temporal;
- comparación mediante MAE y métricas complementarias;
- predicciones para un horizonte de 16 días;
- dashboard en Power BI para comparar ventas reales y previstas.

El histórico termina en 2017, por lo que el resultado se plantea como una demostración metodológica y no como una herramienta actual para Corporación Favorita.
