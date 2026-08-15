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

No pretendo construir un modelo separado para cada tienda en el MVP. Estas segmentaciones se utilizarán principalmente para entender heterogeneidad y analizar errores.

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
| Baseline naive semanal | Regla simple | Predice usando las ventas observadas 7 días antes. Es una referencia fácil de entender y razonable si existe patrón semanal. | No utiliza promociones, festivos ni cambios de tendencia. |
| Regresión lineal / Ridge | Modelo interpretable | Permite comprobar cuánto aportan las variables de calendario, promociones y lags con un modelo sencillo. | Puede quedarse corta ante relaciones no lineales. |
| LightGBM o XGBoost | Árboles de gradient boosting | Puede capturar relaciones no lineales e interacciones entre tienda, familia, calendario, promociones y retardos. | Es menos interpretable y requiere más ajuste y recursos. |

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
| `onpromotion` | Número de productos en promoción | Numérica | Contexto comercial conocido |
| `store_type`, `cluster` | Características de tienda | Categórica | Contexto del establecimiento |
| `year`, `month`, `day_of_week`, `week_of_year` | Variables de calendario | Numérica/categórica | Estacionalidad |
| `is_weekend` | Fin de semana | Booleana | Patrón semanal |
| `is_holiday` | Festivo aplicable | Booleana | Efecto de calendario |
| `sales_lag_7` | Ventas de hace 7 días | Numérica | Patrón semanal |
| `sales_lag_14` | Ventas de hace 14 días | Numérica | Memoria temporal |
| `sales_lag_28` | Ventas de hace 28 días | Numérica | Tendencia reciente |
| `sales_rolling_mean_7` | Media de los 7 días anteriores | Numérica | Nivel reciente de ventas |
| `sales_rolling_mean_28` | Media de los 28 días anteriores | Numérica | Tendencia suavizada |

Las variables temporales se calcularán siempre con `shift` antes de la media móvil, de forma que la fila actual no utilice su propio valor de ventas.

### Variables que no se usarán directamente

- `transactions` del mismo día: no está disponible para las fechas futuras, por lo que produciría fuga de información si se usa directamente.
- `sales` actual: es la variable objetivo y nunca puede formar parte de las entradas de esa misma fila.
- identificadores como `source_id`: se conservarán para trazabilidad, pero no deberían aportar información predictiva.
- información de inventario, costes o margen: no existe en el dataset.

El precio del petróleo se podrá probar como variable adicional, pero no será imprescindible. Si su aportación es mínima, se eliminará para simplificar el modelo.

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
3. Crear las variables temporales utilizando solo información pasada.
4. Reservar los periodos de validación y test antes de ajustar modelos.
5. Construir el baseline naive semanal.
6. Entrenar un modelo sencillo, inicialmente Ridge o regresión lineal.
7. Entrenar un modelo de gradient boosting si el tiempo y los recursos lo permiten.
8. Comparar los modelos con la misma separación temporal y las mismas métricas.
9. Analizar errores globales, por periodo, tienda y familia.
10. Seleccionar el modelo que ofrezca el mejor equilibrio entre error, estabilidad, complejidad y facilidad de integración.

### Preprocesamiento

- Los nulos de variables auxiliares se tratarán según las reglas de la entrega 3.
- Las variables categóricas se codificarán de forma compatible con el modelo elegido.
- Para Ridge se aplicará codificación de categóricas y, si es necesario, escalado de variables numéricas.
- Para LightGBM/XGBoost no será necesario escalar las variables numéricas.
- Los valores extremos de ventas no se eliminarán automáticamente.
- Se podrá probar `log1p(sales)` únicamente si mejora la estabilidad y después se transforma la predicción de nuevo a la escala original.

### Regla de selección

No seleccionaré un modelo únicamente porque tenga la menor métrica en una única partición. El modelo final deberá:

- superar al baseline en MAE de forma consistente;
- no mostrar un deterioro grande entre validaciones temporales;
- funcionar razonablemente en distintas tiendas y familias;
- poder reproducirse con el pipeline del proyecto;
- generar las 28.512 predicciones del horizonte final sin errores.

Si dos modelos tienen resultados muy parecidos, se elegirá el más sencillo.

## 7. Estrategia de validación y evaluación

La validación será **temporal**, nunca aleatoria. Una división aleatoria mezclaría pasado y futuro y podría producir una evaluación demasiado optimista.

### Separación de datos

La primera propuesta será:

- **Train:** histórico anterior al último bloque de 32 días reservado.
- **Validación:** 16 días inmediatamente anteriores al test final.
- **Test interno:** los últimos 16 días del histórico con `sales` disponible.

Además, si el coste computacional lo permite, realizaré un backtesting sencillo con varios bloques consecutivos de 16 días. Así podré comprobar si la mejora se mantiene en más de un periodo.

El `test.csv` oficial de Kaggle se utilizará después para generar las predicciones finales, pero como no incluye `sales`, no sirve por sí solo para calcular métricas locales.

### Prevención de data leakage

Para evitar fuga temporal:

- los lags y medias móviles se calcularán solo con fechas anteriores;
- el escalado o cualquier transformación aprendida se ajustará solo con train;
- no se utilizarán transacciones del mismo día futuro;
- no se calcularán medias globales usando información de validación o test;
- las particiones respetarán siempre el orden cronológico.

### Métricas

La métrica principal será **MAE (Mean Absolute Error)** porque se interpreta directamente en unidades de ventas y no penaliza tanto los valores extremos como RMSE.

Como métrica secundaria usaré **RMSE** para detectar si existen errores grandes que el MAE puede ocultar.

También revisaré MAE por tienda, familia y periodo. No utilizaré MAPE como métrica principal porque hay ventas iguales a cero y eso genera problemas de interpretación y divisiones por cero.

| Elemento | Decisión prevista | Justificación |
|---|---|---|
| Separación | Split temporal + backtesting de 16 días si es viable | Se parece al uso real y evita mezclar futuro y pasado |
| Métrica principal | MAE | Fácil de interpretar y robusta frente a algunos picos |
| Métrica secundaria | RMSE | Penaliza errores grandes |
| Baseline | Ventas de hace 7 días | Referencia simple con sentido semanal |
| Criterio de aceptación | Mejora mínima del 5% en MAE frente al baseline y resultado estable | Evita aceptar mejoras casi irrelevantes |

El 5% se utilizará como umbral inicial de trabajo. Si al analizar el problema se observa que es demasiado exigente o demasiado bajo, se documentará el cambio y se justificará con los resultados.

### Análisis de errores

Revisaré:

- días con mayor error absoluto;
- tiendas y familias con mayor MAE;
- comportamiento en promociones y festivos;
- periodos con ventas muy altas o muchos ceros;
- diferencia entre baseline y modelo mejorado.

Si un modelo tiene una media buena pero falla mucho en determinados segmentos, se reflejará en el dashboard y en las conclusiones.

## 8. Riesgos y alternativas

### La variable objetivo no representa demanda real

`sales` registra ventas, no demanda insatisfecha. Si hubo una rotura de stock, el dataset no permite saber cuántas unidades se habrían vendido con inventario suficiente. Este es el principal límite conceptual del proyecto. Por eso el resultado se describirá siempre como **predicción de ventas**.

### Data leakage

Es uno de los riesgos técnicos más importantes. Puede aparecer al calcular lags, medias móviles, imputaciones o transformaciones utilizando datos futuros. Se reducirá con un pipeline temporal y comprobaciones explícitas.

### Antigüedad de los datos

El histórico termina en 2017. El proyecto puede demostrar una metodología de forecasting, pero no se presentará como una herramienta actual para Corporación Favorita.

### Volumen y coste computacional

El histórico tiene alrededor de tres millones de filas. Si el entrenamiento completo es demasiado lento, empezaré con una selección de tiendas o familias para validar el pipeline. Después ampliaré el alcance si los recursos lo permiten.

### Series con muchos ceros y comportamiento desigual

Algunas combinaciones tienda-familia pueden tener ventas muy bajas o muchos ceros. Esto puede hacer que una métrica global oculte diferencias importantes. Por eso revisaré el error por segmentos.

### Variables auxiliares con disponibilidad limitada

`transactions` no está disponible para el horizonte futuro. Si una variable no puede conocerse en el momento real de predicción, no se utilizará directamente aunque mejore artificialmente la métrica.

### Alternativa si ningún modelo supera el baseline

Si ningún modelo mejora el baseline de forma consistente, mantendré el baseline como resultado principal del MVP y documentaré que la complejidad añadida no aporta mejora suficiente. Después revisaría:

1. calidad de las variables temporales;
2. longitud de los lags;
3. tratamiento de promociones y festivos;
4. segmentación por familias o tiendas;
5. posible reducción del alcance.

El proyecto seguiría siendo válido porque demostraría mediante una evaluación temporal que un modelo más complejo no siempre mejora una referencia sencilla.

## Resultado esperado de esta entrega

Con esta estrategia queda definido qué se va a predecir, qué análisis se realizará, qué modelos se compararán, qué datos se utilizarán, cómo se evaluarán los resultados y qué riesgos existen. El siguiente paso será implementar el pipeline, construir el baseline y comprobar con validación temporal si un modelo mejorado aporta una ganancia real.
