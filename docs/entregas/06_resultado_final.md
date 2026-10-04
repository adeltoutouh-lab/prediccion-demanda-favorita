# Entrega final — Implementación y resultados

**Alumno:** Adel Toutouh El Bouchti
**Proyecto:** Predicción de ventas en Corporación Favorita
**Fecha:** 4 de octubre de 2026

## Qué he terminado

A partir del diseño de las cinco primeras entregas he implementado el pipeline, el análisis y la evaluación. El proceso genera el horizonte de 16 días y los datos del dashboard. La memoria recoge el problema, las decisiones, los resultados y las limitaciones. Los notebooks contienen salidas de la ejecución real. La presentación resume el proyecto en unos cinco minutos.

| Requisito de las entregas | Implementación y evidencia |
|---|---|
| Problema, usuario y alcance | Memoria, secciones 1 y 10; responsable de tienda, previsión de ventas, sin pedidos automáticos |
| Fuentes, claves y capas raw/processed/gold | `src/data.py`, `data/diccionario.md`, manifests y checksums |
| Calidad y EDA | `scripts/eda.py`, notebook 01, tablas de `outputs/eda/` y figuras |
| Baseline y modelos justificados | Baseline semanal, Ridge y dos configuraciones de LightGBM; `src/models.py`, `config.json` |
| Validación temporal y prevención de fuga | Tres validaciones de 16 días y un test interno; `src/features.py`, `src/pipeline.py`, ocho pruebas |
| Comparación y errores por segmentos | MAE, RMSE, WAPE, RMSLE, sesgo; CSV por horizonte, familia, tienda y promoción |
| Previsión final | `outputs/submission.csv`, 28.512 filas; tres modelos en `powerbi/data/predicciones.csv` |
| Frontal, filtros, contexto y exportación | Proyecto PBIP, páginas Previsión/Evaluación y `powerbi/README.md` |
| Reproducción y documentación | README, paquetes fijados, scripts, memoria, presentación y guion |

## Resultado y decisión

LightGBM de 63 hojas mejora el MAE del baseline en los tres bloques de validación: 23,48%, 25,73% y 13,34%. Se selecciona antes de medir el test interno. En ese test su MAE es 75,85 frente a 96,54 del baseline, una reducción del 21,43%; WAPE pasa de 20,67% a 16,24%. Esta mejora no se interpreta como ahorro ni como una garantía para otras fechas.

Después de evaluar, se entrena de nuevo con una ventana de 730 días hasta el 15/08/2017 para prever el 16–31/08/2017. Las etiquetas del test oficial de Kaggle no están disponibles. Las métricas que muestro en la entrega pertenecen al test interno independiente.

## Ajustes respecto al diseño inicial

- Las variables históricas se calculan por corte y durante la recursión; no se guardan de forma fija en gold. Así puedo demostrar qué información había disponible en cada predicción.
- Transacciones y petróleo se conservan como contexto en las capas de datos, pero no entran en el modelo. El petróleo futuro en gold representa el último valor histórico, no el realizado.
- Se utiliza una copia pública del conjunto, fijada por commit y hash. No afirmo que sus checksums se hayan certificado contra una descarga autenticada de Kaggle.
- Las fechas ausentes se conservan como desconocidas. Los nulos de variables históricas se codifican con indicador de falta de historia para los modelos; las ventas originales se mantienen.
- Se entrega Power BI como PBIP/PBIR, con datos y modelo semántico, para poder revisar los archivos en Git. El mockup de la entrega 5 sigue siendo el diseño previo y sus valores no son resultados del modelo.

## Comprobaciones y límite de la entrega

Las ocho pruebas de calidad y validación temporal han pasado. Los dos notebooks se han ejecutado con IPython en proceso porque el entorno no permite sockets de kernel; tienen resultados y comprobaciones guardados. El informe de Power BI ha pasado los esquemas oficiales, las referencias del modelo, las claves de sus datos y la reconciliación de MAE, RMSE y WAPE.

Queda una comprobación que no puede hacerse en este entorno: **abrir y actualizar el PBIP en Power BI Desktop para comprobar la interfaz y las medidas dentro de Desktop**. Las instrucciones están en `powerbi/README.md`. La memoria y la presentación indican esta limitación, sin presentar una captura como si el informe ya se hubiese abierto.
