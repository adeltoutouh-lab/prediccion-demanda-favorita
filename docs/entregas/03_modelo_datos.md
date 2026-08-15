# Entrega 3 - Diseño del modelo de datos y capa gold

**Alumno:** Adel Toutouh El Bouchti  
**Proyecto:** Data Science / IA  
**Idea seleccionada:** Predicción de demanda para Corporación Favorita

## 1. Resumen de la idea y datos del proyecto

El proyecto busca predecir las ventas diarias de las distintas familias de productos en las tiendas de Corporación Favorita. El problema que se intenta resolver es la dificultad de saber con antelación cuánto se va a vender en cada tienda. Una previsión demasiado baja puede causar falta de productos y una previsión demasiado alta puede generar exceso de inventario. La solución que quiero construir es un modelo de predicción acompañado de un dashboard en Power BI para consultar el histórico, las previsiones y los errores del modelo.

La fuente principal será el dataset **Store Sales - Time Series Forecasting** publicado en Kaggle. Los archivos principales son `train.csv` y `test.csv`. El primero contiene el histórico de ventas y el segundo contiene el periodo de 16 días que se debe predecir. También se utilizarán `stores.csv`, `transactions.csv`, `holidays_events.csv` y `oil.csv`.

Cada fuente aporta una parte diferente de información:

- `train.csv`: fecha, tienda, familia de producto, ventas y productos en promoción.
- `test.csv`: mismas combinaciones de fecha, tienda y familia, pero sin la variable de ventas.
- `stores.csv`: ciudad, provincia, tipo y clúster de cada tienda.
- `transactions.csv`: número de transacciones realizadas por tienda y día.
- `holidays_events.csv`: festivos y eventos nacionales, regionales o locales.
- `oil.csv`: precio diario del petróleo.

## 2. Tecnología o formato de almacenamiento elegido

Voy a utilizar una combinación de **CSV y Parquet**.

Los ficheros originales se conservarán en CSV dentro de la capa `raw`, porque es el formato en el que se descargan desde Kaggle. No modificaré estos archivos, ya que quiero mantener una copia exacta de la fuente por si tengo que repetir el proceso de limpieza desde el principio.

Para las capas `processed` y `gold` utilizaré principalmente Parquet. El motivo es que el conjunto de entrenamiento tiene alrededor de tres millones de filas y Parquet ocupa menos espacio, conserva los tipos de datos y suele cargarse más rápido con Python y pandas. Además, permite guardar columnas numéricas con tipos más pequeños, por ejemplo `int16` o `float32`, para reducir el uso de memoria.

No voy a utilizar PostgreSQL o MySQL en esta fase porque añadiría complejidad sin ser necesario para el MVP. Los datos son tabulares y el proyecto se puede desarrollar de forma reproducible con Python y archivos Parquet. Si Power BI diera algún problema al leer Parquet, exportaría una versión CSV de las columnas que necesite para el dashboard.

## 3. Estructura de capas de datos

La estructura prevista será la siguiente:

```text
data/
|-- raw/
|   |-- train.csv
|   |-- test.csv
|   |-- stores.csv
|   |-- transactions.csv
|   |-- holidays_events.csv
|   `-- oil.csv
|-- processed/
|   |-- sales_history_clean.parquet
|   |-- forecast_base_clean.parquet
|   |-- stores_clean.parquet
|   |-- transactions_clean.parquet
|   |-- holidays_store_date.parquet
|   `-- oil_daily.parquet
`-- gold/
    |-- gold_sales_history.parquet
    `-- gold_forecast_horizon.parquet
```

### Capa raw

Contendrá los archivos originales descargados de Kaggle. Solo se descomprimirán y se guardarán con su nombre original. No se cambiarán columnas ni valores.

### Capa processed

Contendrá los datos después de aplicar las primeras transformaciones:

- conversión de fechas;
- corrección de tipos de datos;
- revisión de duplicados;
- tratamiento de valores nulos;
- cambio de nombres de columnas ambiguas;
- preparación de festivos por tienda y fecha;
- comprobación de claves antes de realizar los cruces.

En esta capa las tablas seguirán estando separadas. Esto permitirá revisar cada fuente antes de unirla con las demás.

### Capa gold

La capa gold contendrá los datasets finales que consumirán el análisis exploratorio, el modelo y el dashboard. Las tablas estarán unidas y tendrán nombres de columnas claros. También incluirán variables de calendario y marcas que indiquen si un valor ha sido imputado.

## 4. Definición de la capa gold

La capa gold estará formada inicialmente por dos datasets. He separado el histórico y el horizonte de predicción porque el histórico contiene la variable `sales`, mientras que las fechas futuras no la contienen.

### 4.1 `gold_sales_history.parquet`

| Elemento | Definición |
|---|---|
| Descripción | Histórico limpio y enriquecido de ventas |
| Granularidad | Una fila por fecha, tienda y familia de producto |
| Registros esperados | Aproximadamente 3 millones |
| Identificador | Clave compuesta: `date`, `store_nbr`, `family` |
| Variable objetivo | `sales` |
| Uso posterior | EDA, entrenamiento del modelo, validación temporal y dashboard |

Campos principales:

- identificadores de fecha, tienda y familia;
- ventas y promociones;
- ciudad, provincia, tipo y clúster de tienda;
- transacciones históricas;
- precio del petróleo;
- información de festivos;
- variables de calendario;
- indicadores de valores imputados.

Las variables de retardo y medias móviles se crearán respetando el orden temporal. Se podrán guardar en este dataset cuando el pipeline esté validado. Las primeras fechas tendrán valores nulos en estas variables porque todavía no existe histórico anterior suficiente. Para entrenar el modelo se excluirán las filas que no tengan el histórico mínimo necesario.

### 4.2 `gold_forecast_horizon.parquet`

| Elemento | Definición |
|---|---|
| Descripción | Datos preparados para las fechas que se deben predecir |
| Granularidad | Una fila por fecha futura, tienda y familia de producto |
| Registros esperados | 28.512 filas |
| Identificador | Clave compuesta: `date`, `store_nbr`, `family` |
| Variable objetivo | No disponible; será el valor que debe generar el modelo |
| Uso posterior | Generación de predicciones y presentación final |

Este dataset tendrá las variables conocidas antes de realizar la predicción, como tienda, familia, promociones, calendario, festivos y precio del petróleo. No utilizaré las transacciones del mismo día como variable predictora porque no están disponibles para el periodo futuro. Como alternativa, se podrán utilizar retardos de transacciones calculados únicamente con fechas anteriores.

### Contrato básico de la capa gold

Para considerar válida la capa gold se tendrán que cumplir estas reglas:

1. No puede haber más de una fila para la misma combinación de `date`, `store_nbr` y `family`.
2. `date`, `store_nbr`, `family` y `onpromotion` no pueden ser nulos.
3. `sales` no puede ser negativo en el histórico.
4. `onpromotion`, `transactions` y `cluster` no pueden tener valores negativos.
5. Después de los joins, el número de filas debe mantenerse. Si aumenta, significará que alguna tabla auxiliar tiene claves duplicadas.
6. Las columnas del histórico y del horizonte futuro deben tener nombres y tipos compatibles.

## 5. Relaciones entre los datos

La tabla principal será la formada por `train.csv` y `test.csv`. Las demás fuentes se relacionarán con ella mediante las siguientes claves:

```text
stores.store_nbr
        1
        |
        N
sales.store_nbr

transactions.(date, store_nbr)
        1
        |
        N
sales.(date, store_nbr, family)

oil.date
        1
        |
        N
sales.date

holidays_store_date.(date, store_nbr)
        1
        |
        N
sales.(date, store_nbr, family)
```

Las relaciones previstas son:

- `stores` con ventas: relación 1:N mediante `store_nbr`.
- `transactions` con ventas: relación 1:N mediante `date` y `store_nbr`, porque una cifra diaria de transacciones se repite para todas las familias de esa tienda.
- `oil` con ventas: relación 1:N mediante `date`.
- festivos con ventas: relación 1:N mediante `date` y `store_nbr`, después de preparar una tabla intermedia.

La tabla de festivos es la que requiere más trabajo. Un festivo nacional afecta a todas las tiendas, uno regional solo a las tiendas de una provincia y uno local solo a las tiendas de una ciudad. Además, puede haber más de un evento en la misma fecha. Para evitar duplicar ventas, primero crearé `holidays_store_date.parquet`, con una única fila por fecha y tienda. Si existen varios eventos, se combinarán en indicadores y en una descripción agrupada.

También renombraré columnas que tienen el mismo nombre en distintas tablas. Por ejemplo, `stores.type` pasará a llamarse `store_type` y `holidays_events.type` pasará a llamarse `holiday_type`.

## 6. Diccionario de datos inicial

| Campo | Descripción | Tipo esperado | Fuente | Obligatorio | Observaciones |
|---|---|---|---|---|---|
| `source_id` | Identificador original del registro | int64 | train/test | Sí | Se conserva para trazabilidad |
| `date` | Fecha del registro | date | train/test | Sí | Formato `YYYY-MM-DD` |
| `store_nbr` | Número de tienda | int16 | train/test | Sí | Parte de la clave principal |
| `family` | Familia de producto | category/string | train/test | Sí | Parte de la clave principal |
| `sales` | Ventas diarias | float32 | train | Sí en histórico | Puede contener decimales |
| `onpromotion` | Productos de la familia en promoción | int32 | train/test | Sí | No debe ser negativo |
| `city` | Ciudad de la tienda | category/string | stores | Sí | Se usa para festivos locales |
| `state` | Provincia o región de la tienda | category/string | stores | Sí | Se usa para festivos regionales |
| `store_type` | Tipo de establecimiento | category/string | stores | Sí | Renombrado desde `type` |
| `cluster` | Grupo de tiendas similares | int8 | stores | Sí | Valores positivos |
| `transactions` | Transacciones diarias de la tienda | int32 nullable | transactions | No | Solo se conoce para el histórico |
| `transactions_missing` | Indica si faltaba la cifra de transacciones | bool | derivada | Sí | Evita ocultar la imputación |
| `oil_price` | Precio diario del petróleo | float32 | oil | No | Renombrado desde `dcoilwtico` |
| `oil_price_imputed` | Indica si el precio fue rellenado | bool | derivada | Sí | Útil para revisar calidad |
| `is_holiday` | Indica si existe festivo aplicable a la tienda | bool | holidays | Sí | Nacional, regional o local |
| `holiday_type` | Tipo de festivo o evento | category/string | holidays | No | Puede agrupar varios eventos |
| `holiday_description` | Descripción del evento | string | holidays | No | Texto agrupado si hay varios |
| `is_transferred` | Indica si el festivo fue trasladado | bool | holidays | Sí | Se revisará su fecha efectiva |
| `year` | Año | int16 | derivada | Sí | Variable de calendario |
| `month` | Mes | int8 | derivada | Sí | Valores de 1 a 12 |
| `day_of_week` | Día de la semana | int8 | derivada | Sí | Valores de 0 a 6 |
| `week_of_year` | Semana del año | int8 | derivada | Sí | Variable estacional |
| `is_weekend` | Indica sábado o domingo | bool | derivada | Sí | Variable de calendario |
| `is_month_start` | Inicio de mes | bool | derivada | Sí | Variable de calendario |
| `is_month_end` | Final de mes | bool | derivada | Sí | Variable de calendario |
| `dataset_split` | Indica histórico o periodo futuro | category/string | derivada | Sí | Valores `train` o `forecast` |

Cuando se añadan variables temporales para el modelo, se utilizarán nombres como `sales_lag_7`, `sales_lag_14`, `sales_lag_28`, `sales_rolling_mean_7` y `sales_rolling_mean_28`.

## 7. Problemas de calidad esperados

Los principales problemas que espero encontrar son los siguientes:

### Valores nulos

El precio del petróleo puede tener fechas sin valor, sobre todo en fines de semana o días sin cotización. También pueden faltar transacciones para determinadas combinaciones de tienda y fecha. Estos casos no se deben rellenar sin dejar una marca de que se ha realizado una imputación.

### Duplicados

En `train.csv` y `test.csv` la combinación de fecha, tienda y familia debería ser única. También debería existir una sola fila de transacciones por fecha y tienda. Haré una comprobación antes de los joins, porque un duplicado en una tabla auxiliar podría multiplicar el número de registros del histórico.

### Festivos con distinta cobertura

Los festivos tienen niveles nacional, regional y local. No se pueden unir solo mediante la fecha, porque eso aplicaría un festivo local a todas las tiendas. También puede haber varios eventos el mismo día y existen festivos trasladados.

### Columnas con nombres repetidos

Las tablas `stores` y `holidays_events` utilizan nombres como `type`. Si se hace el join sin renombrarlas, el resultado puede ser confuso. Por eso se usarán `store_type` y `holiday_type`.

### Ventas extremas

Pueden aparecer picos reales por promociones, festivos o eventos. No eliminaré automáticamente los valores altos como si fueran errores. Primero comprobaré si coinciden con una promoción o un evento. Para el modelo se podrá aplicar `log1p(sales)` o utilizar métricas menos sensibles a los valores extremos.

### Ceros en ventas

Una venta igual a cero puede significar que no hubo demanda, que la tienda estaba cerrada o que existió una rotura de stock. Como el dataset no contiene inventario, no será posible diferenciar todos estos casos. Los ceros se conservarán porque son observaciones válidas.

### Datos desactualizados

El histórico termina en 2017. Esto no impide realizar el proyecto académico, pero limita la aplicación de las conclusiones al funcionamiento actual de la empresa.

### Variables no disponibles en el futuro

Las transacciones del mismo día no están disponibles para el horizonte que se quiere predecir. Utilizarlas directamente provocaría fuga de información. Solo se usarán para EDA o mediante variables retardadas basadas en fechas anteriores.

## 8. Decisiones de limpieza y transformación previstas

Las decisiones iniciales serán las siguientes:

1. **Conservar los CSV originales.** La capa raw no se modificará.
2. **Normalizar fechas.** Todas las columnas de fecha se convertirán al tipo `date` y al formato `YYYY-MM-DD`.
3. **Optimizar tipos.** Se utilizarán enteros pequeños y `float32` cuando no se pierda información.
4. **Comprobar claves.** Antes de los joins se revisará la unicidad de cada clave esperada.
5. **Gestionar duplicados.** Si aparece un duplicado en una clave que debería ser única, no se eliminará automáticamente. Primero se revisará su causa y el pipeline mostrará un aviso o error.
6. **Preparar festivos.** Se aplicarán según ciudad, provincia o nivel nacional. Los festivos con `transferred=True` no se marcarán como festivo efectivo en su fecha original sin revisar la fila de traslado correspondiente.
7. **Tratar el petróleo.** Se ordenará por fecha y se aplicará un relleno hacia delante con el último precio conocido. Si queda algún nulo al comienzo, se utilizará una medida calculada solo con el periodo de entrenamiento. Se añadirá `oil_price_imputed`.
8. **Tratar transacciones.** Si falta una cifra y las ventas totales de esa tienda y día son cero, se podrá asignar cero. En otros casos se dejará como nulo o se imputará mediante la mediana histórica de la tienda y del día de la semana. Se añadirá `transactions_missing`.
9. **Crear variables de calendario.** Año, mes, semana, día de la semana, fin de semana, inicio y final de mes.
10. **Crear retardos sin fuga temporal.** Los lags y medias móviles se calcularán agrupando por tienda y familia, ordenando por fecha y aplicando primero un `shift`. No se utilizará el valor actual para calcular sus propias variables.
11. **No eliminar outliers de forma automática.** Se estudiarán junto con promociones y festivos.
12. **Separar histórico y futuro.** El modelo se validará con divisiones cronológicas y no con una partición aleatoria.
13. **Controlar el número de filas.** Después de cada join se comprobará que no se hayan duplicado registros.

Un registro se considerará inválido si no tiene fecha, tienda o familia, si presenta ventas negativas, si tiene promociones negativas o si repite una clave que debería ser única sin una explicación válida.

## 9. Riesgos del modelo de datos

La parte más clara del modelo es la granularidad. Cada fila representa una fecha, una tienda y una familia de producto. También está clara la relación con `stores.csv`, porque cada tienda tiene un único registro con sus características.

La parte que genera más incertidumbre es el tratamiento de los festivos y de las variables que no se conocen en el futuro. La tabla que probablemente dará más problemas será `holidays_events.csv`, porque hay que decidir a qué tiendas afecta cada evento y evitar que una fecha con varios eventos duplique las ventas. `transactions.csv` también es una limitación porque no tiene información para el horizonte futuro.

Si no pudiera construir la capa gold completa, simplificaría el modelo utilizando solo ventas, promociones, tienda, familia y variables de calendario. Estas son las variables imprescindibles y están presentes tanto en el histórico como en el periodo futuro. Después añadiría las tablas auxiliares una por una para medir si realmente mejoran el resultado.

Otra opción sería trabajar primero con una selección de tiendas y familias para comprobar todo el pipeline. Cuando el proceso funcionase correctamente, se aplicaría al conjunto completo. Si Parquet causara problemas con alguna herramienta, se mantendría la misma estructura pero se exportaría la capa gold en CSV.

## Referencias

- Kaggle. Store Sales - Time Series Forecasting: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- Kaggle. Datos de la competición: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
- Corporación Favorita: https://www.corporacionfavorita.com/
