# Entrega 4 - Diseño del análisis y estrategia de modelado

**Alumno:** Adel Toutouh El Bouchti  
**Proyecto:** Data Science / IA  
**Proyecto seleccionado:** Predicción de ventas para Corporación Favorita

## Ajuste de alcance respecto a las entregas anteriores

En las primeras entregas utilicé varias veces el término "predicción de demanda". Después de revisar mejor el dataset y el comentario recibido en la entrega 2, voy a acotar el objetivo de forma más precisa: **el modelo predecirá ventas registradas, no demanda real**.

El dataset de Corporación Favorita no contiene inventario disponible ni roturas de stock. Por tanto, una venta baja puede deberse a poca demanda, pero también a que el producto no estuviera disponible. Con estos datos no puedo distinguir ambos casos. Tampoco voy a plantear una optimización automática de inventario ni estimaciones de ahorro, porque faltarían costes, márgenes, caducidad y decisiones reales de compra.

El proyecto se mantiene como un ejercicio de forecasting con datos históricos reales. El resultado servirá para demostrar una metodología reproducible y para apoyar el análisis de planificación, pero no se presentará como una herramienta actualmente operativa en Corporación Favorita, ya que el histórico termina en 2017.

## 1. Problema que se busca resolver

El problema que quiero resolver es estimar las **ventas diarias futuras** para cada combinación de tienda y familia de producto. Actualmente, si solo se mira el histórico sin un modelo, es difícil tener una referencia consistente de cuánto se puede vender en los días siguientes, sobre todo cuando existen promociones, festivos, diferencias entre tiendas y patrones estacionales.

El resultado lo utilizaría un perfil de analista o responsable de planificación como apoyo para revisar previsiones por tienda y familia. No tomaría decisiones de compra automáticamente. La utilidad estaría en disponer de una previsión comparable con las ventas reales y en detectar dónde el modelo se equivoca más.

Para considerar que el proyecto aporta valor, debería cumplir tres condiciones:

1. Generar previsiones para un horizonte de **16 días**, que es el horizonte del conjunto de test de la competición.
2. Mejorar de forma clara un baseline sencillo.
3. Mantener un error razonablemente estable entre distintos periodos, tiendas y familias, sin depender de información futura.

## 2. Análisis de datos planteado y utilidad esperada

El análisis exploratorio estará orientado al problema de forecasting. No quiero hacer gráficos solo por completar el EDA, sino comprobar patrones que después puedan justificar variables o decisiones del modelo.

### Preguntas principales

- ¿Cómo evolucionan las ventas a lo largo del tiempo?
- ¿Existe estacionalidad semanal, mensual o anual?
- ¿Qué familias y tiendas concentran más ventas?
- ¿Hay combinaciones tienda-familia con muchos ceros?
- ¿Qué diferencias aparecen entre días laborables y fines de semana?
- ¿Las promociones coinciden con incrementos de ventas?
- ¿Los festivos y eventos producen cambios visibles?
- ¿Existen periodos con cambios bruscos o valores extremos?
- ¿Qué tiendas o familias presentan mayor variabilidad y pueden ser más difíciles de predecir?

### Análisis descriptivo y temporal

Calcularé ventas totales, medias, medianas, desviaciones y percentiles. También revisaré la distribución de `sales` y el porcentaje de valores cero.

En la parte temporal estudiaré:

- ventas por día y semana;
- media por día de la semana;
- evolución mensual;
- comparación entre años;
- autocorrelación de forma básica mediante retardos de 7, 14 y 28 días;
- medias móviles de 7 y 28 días.

La utilidad será comprobar si tiene sentido utilizar variables de calendario, lags y medias móviles.

### Comparaciones por tienda y familia

Compararé ventas y error posterior del modelo por:

- `store_nbr`;
- `family`;
- `store_type`;
- `cluster`;
- ciudad o provincia cuando sea útil.

No pretendo construir un modelo separado para cada tienda en el MVP. Estas segmentaciones se utilizarán principalmente para entender diferencias y analizar errores.

### Promociones y festivos

Analizaré las ventas medias con y sin promoción, teniendo cuidado de no interpretar la relación como causalidad. También revisaré las fechas con festivos o eventos aplicables a cada tienda.

Una hipótesis razonable es que promociones y festivos ayudan a explicar parte de los picos de ventas. Si no aportan mejora en validación, se podrán retirar del modelo final.

### Visualizaciones para el MVP

El dashboard de Power BI podrá incluir:

- evolución histórica de ventas;
- ventas reales frente a previstas;
- error absoluto por fecha;
- MAE global y por tienda/familia;
- filtros por tienda, familia y periodo;
- comparación de días con y sin promoción;
- tabla o ranking de combinaciones con mayor error.

Estas visualizaciones permitirán interpretar el resultado sin presentar la predicción como una cifra exacta.

## 3. Tipo de modelos que se van a plantear

La tarea será un problema de **forecasting supervisado**, formulado como regresión sobre datos temporales. La variable objetivo es `sales`.

No voy a comparar muchos algoritmos. Para el MVP prefiero tener un baseline claro, un modelo sencillo y un modelo más flexible.

| Alternativa | Tipo | Por qué se plantea | Limitación principal |
|---|---|---|---|
| Baseline naive semanal | Regla simple | Usa como referencia la venta de 7 días antes. En un horizonte de 16 días deberá respetar la misma restricción temporal que el resto de modelos. | No utiliza promociones, festivos ni cambios de tendencia. |
| Regresión lineal / Ridge | Modelo interpretable | Permite comprobar cuánto aportan las variables de calendario, promociones y lags con un modelo sencillo. | Puede quedarse corta ante relaciones no lineales. |
| LightGBM o XGBoost | Árboles de gradient boosting | Puede captar relaciones no lineales e interacciones entre tienda, familia, calendario, promociones y retardos. | Es menos interpretable y requiere más ajuste y recursos. |

Si el modelo avanzado no mejora de forma consistente al baseline y al modelo sencillo, no se seleccionará solo por ser más complejo.

## 4. Datos de entrada del análisis y los modelos

La fuente principal de entrada será la capa gold definida en la entrega 3.

### Dataset principal

**`gold_sales_history.parquet`**

- Granularidad: una fila por `date`, `store_nbr` y `family`.
- Clave principal: combinación de `date`, `store_nbr`, `family`.
- Variable objetivo: `sales`.
- Uso: EDA, entrenamiento y validación temporal.

Para la predicción final se utilizará **`gold_forecast_horizon.parquet`**, con la misma granularidad pero sin `sales`.

### Variables de entrada previstas

| Entrada | Descripción | Tipo | Uso |
|---|---|---|---|
| `store_nbr` | Identificador de tienda | Categórica | Diferencias entre establecimientos |
| `family` | Familia de producto | Categórica | Diferencias entre categorías |
| `onpromotion` | Número de productos en promoción | Numérica | Variable futura conocida en `test.csv` |
| `store_type`, `cluster` | Características de tienda | Categórica | Contexto del establecimiento |
| `year`, `month`, `day_of_week`, `week_of_year` | Variables de calendario | Numérica/categórica | Estacionalidad |
| `is_weekend` | Fin de semana | Booleana | Patrón semanal |
| `is_holiday` | Festivo conocido por calendario | Booleana | Efecto de calendario |
| `sales_lag_7` | Ventas de hace 7 días | Numérica | Patrón semanal |
| `sales_lag_14` | Ventas de hace 14 días | Numérica | Memoria temporal |
| `sales_lag_28` | Ventas de hace 28 días | Numérica | Tendencia reciente |
| `sales_rolling_mean_7` | Media de los 7 valores anteriores disponibles | Numérica | Nivel reciente de ventas |
| `sales_rolling_mean_28` | Media de los 28 valores anteriores disponibles | Numérica | Tendencia suavizada |

### Cómo se calcularán los lags durante los 16 días

Esta parte requiere una precaución especial. No basta con calcular los lags sobre todo el histórico y después separar validación, porque a partir de determinados días del horizonte se podrían estar utilizando ventas reales de días anteriores del propio bloque de 16 días.

Para evitarlo, la simulación de validación funcionará como la predicción final: al comenzar un horizonte de 16 días se fija una **fecha de corte** y las ventas reales de esos 16 días se consideran desconocidas.

La predicción será **recursiva**. Para los primeros días se utilizará el histórico real anterior a la fecha de corte. Cuando un lag o una media móvil necesite un valor que ya cae dentro de los 16 días futuros, se utilizará la predicción generada previamente por el propio modelo, no la venta real que después usaré para evaluar.

Ejemplo: al predecir el día 8, `sales_lag_7` apuntará al día 1 del horizonte. Como ese valor no se conocería en el momento real de lanzar las 16 previsiones, se utilizará la predicción obtenida para el día 1.

Esto puede hacer que el error se acumule a lo largo del horizonte, pero es una limitación real que interesa medir.

### Variables que no se usarán directamente

- `transactions` del mismo día futuro: no aparece en `test.csv` y no se conocería al lanzar la predicción.
- `sales` actual: es la variable objetivo.
- identificadores como `source_id`: se conservan para trazabilidad, pero no deberían aportar información predictiva.
- información de inventario, costes o margen: no existe en el dataset.
- precio del petróleo futuro: aunque existe como fuente auxiliar histórica, no voy a asumir que se conoce con antelación para todo el horizonte. Para el MVP prefiero dejarlo fuera del modelo final.

Antes de utilizar cualquier variable externa comprobaré que realmente estaría disponible en la fecha de corte. Las promociones del periodo futuro sí se pueden utilizar porque están incluidas en `test.csv`; las variables de calendario y las características de tienda también se conocen de antemano.

## 5. Datos de salida y forma de consumo

La salida principal será una predicción numérica de ventas para cada combinación de fecha, tienda y familia.

| Campo de salida | Descripción | Tipo | Uso posterior |
|---|---|---|---|
| `date` | Fecha predicha | date | Trazabilidad temporal |
| `store_nbr` | Tienda | integer | Filtro y unión |
| `family` | Familia de producto | string | Filtro y unión |
| `sales_pred` | Ventas previstas | float | Resultado principal |
| `model_name` | Modelo que genera la predicción | string | Trazabilidad |
| `run_date` | Fecha de ejecución | datetime | Reproducibilidad |

Durante validación se añadirá también `sales_real`, `abs_error` y, si procede, una métrica relativa calculada con cuidado en los casos donde las ventas sean cero.

El formato previsto será Parquet o CSV para conectar con Power BI. En el dashboard el usuario podrá comparar ventas reales y previstas y revisar métricas de error.

No mostraré una supuesta "demanda real" ni una recomendación automática de unidades a comprar, porque esos resultados no se pueden justificar con los datos disponibles.

## 6. Estrategia para diseñar y seleccionar el modelo

El proceso previsto será el siguiente:

1. Cargar `gold_sales_history.parquet` y comprobar claves, fechas, nulos y tipos.
2. Ordenar los datos cronológicamente por tienda y familia.
3. Separar primero los periodos que se usarán para validar.
4. Crear las variables temporales del entrenamiento usando únicamente datos anteriores.
5. Construir el baseline naive semanal.
6. Entrenar un modelo sencillo, inicialmente Ridge o regresión lineal.
7. Entrenar un modelo de gradient boosting si el tiempo y los recursos lo permiten.
8. Simular un horizonte completo de 16 días sin consultar las ventas reales de ese bloque.
9. Comparar los modelos con las mismas restricciones y métricas.
10. Analizar errores globales, por día del horizonte, tienda y familia.
11. Seleccionar el modelo con mejor equilibrio entre error, estabilidad, complejidad y facilidad de integración.

### Preprocesamiento

- Los nulos de variables auxiliares se tratarán según las reglas de la entrega 3.
- Las variables categóricas se codificarán de forma compatible con el modelo elegido.
- Para Ridge se aplicará codificación de categóricas y, si es necesario, escalado de variables numéricas.
- Para LightGBM/XGBoost no será necesario escalar las variables numéricas.
- Los valores extremos de ventas no se eliminarán automáticamente.
- Se podrá probar `log1p(sales)` únicamente si mejora la estabilidad y después se transformará la predicción de nuevo a la escala original.

### Regla de selección

No seleccionaré un modelo únicamente porque tenga la menor métrica en una sola partición. El modelo final deberá:

- superar al baseline en MAE de forma consistente;
- mantener una mejora razonable a lo largo de los 16 días;
- no mostrar un deterioro grande entre validaciones temporales;
- funcionar razonablemente en distintas tiendas y familias;
- poder reproducirse con el pipeline del proyecto;
- generar las 28.512 predicciones del horizonte final sin errores.

Si dos modelos tienen resultados muy parecidos, se elegirá el más sencillo.

## 7. Estrategia de validación y evaluación

La validación será **temporal**, nunca aleatoria. Además, reproducirá la forma en la que se generarán las 16 predicciones finales.

### Separación de datos

La primera propuesta será:

- **Train:** histórico anterior al último bloque de 32 días reservado.
- **Validación:** primer bloque de 16 días reservados.
- **Test interno:** últimos 16 días del histórico con `sales` disponible.

Para cada bloque se fijará una fecha de corte. Desde ese momento se ocultarán las ventas reales de los siguientes 16 días y se generarán las predicciones de manera recursiva. Solo cuando estén generadas las 16 predicciones se compararán con las ventas reales.

Si el coste computacional lo permite, repetiré el mismo proceso con varios bloques anteriores de 16 días como backtesting. Así se podrá comprobar si el resultado depende demasiado de un único periodo.

El `test.csv` oficial de Kaggle se utilizará después para generar las predicciones finales. Como no incluye `sales`, no sirve para calcular métricas locales.

### Mismo criterio para el baseline

El baseline tendrá exactamente la misma restricción. No se le permitirá consultar ventas reales del horizonte que tampoco conocería el modelo.

En el baseline semanal, cuando el valor de hace 7 días quede dentro del propio bloque de predicción, se reutilizará la predicción que el baseline generó para ese día. Así la comparación será más justa.

### Prevención de data leakage

Para evitar fuga temporal:

- se separará cada bloque de validación antes de construir las variables de ese horizonte;
- las ventas reales de los 16 días permanecerán ocultas hasta terminar la predicción;
- los lags y medias móviles usarán únicamente histórico previo o predicciones anteriores del propio horizonte;
- el escalado o cualquier transformación aprendida se ajustará solo con train;
- no se utilizarán transacciones futuras ni otras variables que no estuvieran disponibles en la fecha de corte;
- no se calcularán medias globales utilizando validación o test.

### Métricas

La métrica principal será **MAE (Mean Absolute Error)** porque se interpreta directamente en unidades de ventas y no penaliza tanto los valores extremos como RMSE.

Como métrica secundaria usaré **RMSE** para detectar si existen errores grandes que el MAE puede ocultar.

Además del resultado global, revisaré el MAE por **día del horizonte (1 a 16)**. Esto permitirá ver si la predicción recursiva pierde precisión a medida que se aleja de la fecha de corte.

También revisaré MAE por tienda, familia y periodo. No utilizaré MAPE como métrica principal porque hay ventas iguales a cero y eso genera problemas de interpretación y divisiones por cero.

| Elemento | Decisión prevista | Justificación |
|---|---|---|
| Separación | Split temporal + bloques de 16 días | Reproduce el horizonte real |
| Generación | Predicción recursiva | Evita usar ventas reales del propio horizonte |
| Métrica principal | MAE | Fácil de interpretar |
| Métrica secundaria | RMSE | Ayuda a detectar errores grandes |
| Baseline | Naive semanal recursivo | Se compara bajo la misma restricción que el modelo |
| Criterio de aceptación | Mejora mínima inicial del 5% en MAE frente al baseline y resultado estable | Evita aceptar mejoras casi irrelevantes |

El 5% se utilizará como umbral inicial de trabajo. Si al analizar los resultados se observa que no es un umbral razonable, se documentará cualquier cambio.

### Análisis de errores

Revisaré:

- error por día del horizonte;
- días con mayor error absoluto;
- tiendas y familias con mayor MAE;
- comportamiento en promociones y festivos;
- periodos con ventas muy altas o muchos ceros;
- diferencia entre baseline y modelo mejorado.

Si el error aumenta bastante en los últimos días del horizonte, será una señal de acumulación de error de la estrategia recursiva y se indicará como limitación.

## 8. Riesgos y alternativas

### La variable objetivo no representa demanda real

`sales` registra ventas, no demanda insatisfecha. Si hubo una rotura de stock, el dataset no permite saber cuántas unidades se habrían vendido con inventario suficiente. Por eso el resultado se describirá siempre como **predicción de ventas**.

### Data leakage dentro del horizonte

Es uno de los riesgos técnicos más importantes. El problema no aparece solo al separar train y test: también puede aparecer dentro de los 16 días si se utilizan como lags las ventas reales de días anteriores del propio horizonte. La simulación recursiva evitará este problema.

### Acumulación de error

Al reutilizar predicciones anteriores para construir lags y medias móviles, un error de los primeros días puede afectar a días posteriores. Por eso se medirá el error por posición dentro del horizonte y no únicamente con una métrica global.

### Variables futuras no conocidas

Solo se utilizarán variables que puedan conocerse en la fecha de corte. `onpromotion` está disponible en el test de Kaggle y las variables de calendario son conocidas. No se utilizarán directamente `transactions` futuras ni se asumirá que se conoce el precio futuro del petróleo.

### Antigüedad de los datos

El histórico termina en 2017. El proyecto puede demostrar una metodología de forecasting, pero no se presentará como una herramienta actual para Corporación Favorita.

### Volumen y coste computacional

El histórico tiene alrededor de tres millones de filas. Si el entrenamiento completo es demasiado lento, empezaré con una selección de tiendas o familias para validar el pipeline. Después ampliaré el alcance si los recursos lo permiten.

### Alternativa si ningún modelo supera el baseline

Si ningún modelo mejora el baseline de forma consistente, mantendré el baseline como resultado principal del MVP y documentaré que la complejidad añadida no aporta mejora suficiente. Después revisaría las variables temporales, el tratamiento de promociones y festivos y el alcance del modelo.

## Resultado esperado de esta entrega

Con esta estrategia queda definido qué se va a predecir, qué análisis se realizará, qué modelos se compararán, qué datos se utilizarán y cómo se evitará utilizar información del propio horizonte que no existiría en el momento real de predicción. El siguiente paso será implementar el pipeline y comprobar con bloques completos de 16 días si un modelo mejorado aporta una ganancia real.