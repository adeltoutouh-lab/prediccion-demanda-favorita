# Guion de defensa — cinco minutos

Adel Toutouh El Bouchti. Seis diapositivas. Los tiempos son orientativos: ensaya con un cronómetro y explica las cifras principales sin leer todas las tablas.

## 1. Objetivo y alcance (0:00–0:35)

Mi trabajo parte de una pregunta concreta: cómo preparar una previsión de ventas que ayude a un responsable de tienda a revisar las próximas dos semanas. He utilizado el dataset de Corporación Favorita y he construido un proceso que va desde los CSV hasta las previsiones y el proyecto de Power BI. El objetivo es predecir ventas registradas por tienda y familia. No hablo de demanda real porque no tengo información de stock ni de ventas que se hayan perdido. Todo el ejemplo se sitúa en agosto de 2017.

## 2. Datos y análisis (0:35–1:20)

El histórico contiene tres millones de filas y combina 54 tiendas con 33 familias, lo que da 1.782 series. He comprobado las claves, las fechas y los valores del objetivo. No hay duplicados en la clave, pero faltan cuatro días completos, todos en Navidad, y un 31,30% de las ventas son cero. Por eso he mantenido un calendario diario y he dejado como desconocidas las fechas sin datos. La gráfica muestra un patrón semanal con ventas medias mayores durante el fin de semana. Eso justifica empezar por un baseline semanal.

## 3. Modelos comparados (1:20–2:10)

He comparado un baseline, Ridge y LightGBM. El baseline repite la venta de siete días antes y es una referencia fácil de explicar. Ridge utiliza una relación lineal con regularización. LightGBM permite combinar variables con relaciones no lineales. Trabajo con 27 variables de calendario, promociones, tienda y ventas anteriores. No utilizo las transacciones ni el petróleo del día que estoy prediciendo. Para controlar el tamaño del experimento solo comparo dos configuraciones de LightGBM y una de Ridge. Los modelos comparten una ventana de 730 días y el mismo horizonte.

## 4. Validación temporal (2:10–3:05)

He reservado tres bloques para seleccionar el modelo y un cuarto como test interno. Cada bloque dura 16 días completos. El punto más importante es que, al avanzar dentro del horizonte, cada modelo usa sus propias predicciones. Por ejemplo, desde el octavo día el baseline vuelve a utilizar lo que predijo en la semana anterior, no las ventas reales que ocurrieron después del corte. LightGBM de 63 hojas mejora al baseline en los tres bloques de validación. La elección queda fijada antes del test interno. Así evito elegir el modelo con la misma información con la que presento su resultado final.

## 5. Resultado del test (3:05–4:05)

En el test interno, del 31 de julio al 15 de agosto, el MAE del baseline es 96,54 y el de LightGBM es 75,85. La reducción es del 21,43%. El WAPE baja de 20,67% a 16,24% y el RMSLE también mejora. Ridge queda cerca del baseline en MAE y tiene un RMSLE claramente peor, por lo que no lo escogería para este caso. La mejora del modelo elegido aparece en las 33 familias y en 51 de las 54 tiendas; las otras tres requieren revisión. No presento estos porcentajes como ahorro económico. Son errores de previsión medidos con datos reales de un bloque histórico.

## 6. Entrega y límites (4:05–5:00)

El resultado final incluye 28.512 previsiones para los 16 días siguientes y un proyecto nativo de Power BI con dos páginas. La primera permite filtrar por tienda, familia y modelo y exportar las previsiones. La segunda muestra los errores del test interno. He comprobado la estructura del informe y que sus CSV reproducen las métricas. El proyecto se ha abierto en Power BI Desktop y se entrega también como PBIX con datos. Queda confirmar el refresco explícito y la interacción completa. Como siguiente paso ampliaría el backtesting a otros periodos y validaría el dashboard con un usuario. Con datos de inventario también podría estudiar roturas de stock, pero con este dataset el alcance sigue siendo ventas.

## Preguntas que conviene preparar

**¿Por qué no llamarlo demanda?** Porque solo observo lo vendido. Sin stock no puedo saber qué parte de la demanda quedó sin atender.

**¿Qué impide la fuga de información?** Los lags excluyen el día objetivo y el forecast solo recibe ventas anteriores al corte. Cada modelo alimenta el horizonte con sus propias predicciones. Las pruebas comprueban especialmente el baseline desde el día 8.

**¿Por qué no elegir Ridge si mejora el RMSE del baseline?** La decisión está fijada por MAE en tres validaciones. LightGBM también mejora WAPE y RMSLE en el test; Ridge presenta un RMSLE claramente peor.

**¿Las promociones futuras eran conocidas?** Lo asumo porque el concurso las proporciona. No tengo snapshots históricos para verificar la fecha de planificación; lo explico como limitación del backtest.

**¿Qué representan las 28.512 filas?** Son 54 tiendas × 33 familias × 16 días. El submission lleva una previsión por fila del test oficial; en Power BI hay además previsiones de los tres modelos.

**¿Está probado el dashboard en Desktop?** El proyecto se ha abierto en Power BI Desktop y se ha guardado el PBIX. La captura confirma la carga de datos y las métricas; el refresco explícito y la interacción completa siguen pendientes.

**¿Qué ampliaría después?** Evaluaría más cortes en distintas épocas y validaría el flujo con un usuario. Necesitaría datos de stock para convertirlo en una herramienta de inventario.
