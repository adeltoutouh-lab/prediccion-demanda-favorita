# Diccionario y contratos de datos

La clave del histórico gold y del horizonte gold es **`date`, `store_nbr`, `family`**. Una fila representa la venta agregada de una familia en una tienda en una fecha. La unicidad, la cobertura y las columnas obligatorias se comprueban en `src/data.py`.

| Campo gold | Tipo lógico | Significado y disponibilidad |
|---|---|---|
| id | Entero | Identificador de la fila original; permite recuperar el orden del submission |
| date | Fecha | Día del registro o de la previsión |
| store_nbr | Entero | Tienda; relación con `stores` |
| family | Texto | Familia de productos |
| sales | Real, ≥0 | Ventas registradas; solo en histórico. Puede incluir peso y valores fraccionarios |
| onpromotion | Entero, ≥0 | Número de artículos promocionados en la familia; disponible en train y test |
| city, state | Texto | Ciudad y provincia de la tienda |
| store_type | Texto | Tipo de tienda; renombrado desde `type` |
| cluster | Entero | Clúster de tienda del catálogo |
| is_holiday | 0/1 | Festivo efectivo para la tienda; no activa el festivo original transferido |
| is_event | 0/1 | Evento que afecta a la tienda |
| is_work_day | 0/1 | Día de trabajo indicado por el calendario |
| is_transferred | 0/1 | Presencia de un evento marcado como transferido |
| holiday_type | Texto | Tipos de eventos coincidentes separados por `;`; vacío si no hay evento |
| holiday_description | Texto | Descripciones de eventos coincidentes |
| transactions | Real o nulo | Transacciones históricas por tienda y día. No se usa en el modelo |
| transactions_missing | Booleano | Falta de observación; no se imputa como cero. Solo histórico |
| oil_price | Real o nulo | Petróleo, con último valor observado en fechas históricas sin cotización. No se usa en el modelo |
| oil_price_imputed | Booleano | Se ha rellenado una fecha sin cotización original |
| oil_price_policy | Texto | Solo horizonte: `last_observed_at_cutoff`; no toma el petróleo realizado futuro |
| day_of_week | Entero 0–6 | Lunes a domingo |
| month | Entero 1–12 | Mes |
| is_payday | 0/1 | Indicador de día 15 o último día del mes; aproximación, no nómina observada |

No se rellenan hacia atrás los nulos iniciales del petróleo. El histórico conserva los ceros, los valores altos y los nulos de contexto. `sales`, fecha, tienda, familia y promoción no pueden ser nulos en train; no se admiten ventas negativas ni claves duplicadas.

## Variables para los modelos

Son 27 columnas, descritas por nombre en `outputs/run_metadata.json`. Los códigos categóricos proceden de un catálogo fijo de las tiendas y familias disponibles, no de estadísticas calculadas con ventas futuras.

| Grupo | Variables |
|---|---|
| Categorías | store_nbr, family_code, series_code, city_code, state_code, type_code, cluster, day_of_week, month |
| Calendario y contexto | day_index, year_sin, year_cos, is_payday, is_holiday, is_event, is_work_day |
| Promociones | promotion_log = log(1 + onpromotion), is_promotion |
| Historia anterior | lag_1, lag_7, lag_14, lag_28, mean_7, mean_28, std_28, weekday_mean_4 |
| Disponibilidad | missing_lags: indicador de retardos desconocidos |

Los retardos y las medias excluyen el día que se predice. Los nulos de variables históricas se convierten a cero para suministrar una matriz numérica, junto con el indicador de falta de historia; esto no modifica las ventas originales. Las predicciones de días anteriores alimentan las variables del horizonte. Cada modelo mantiene un estado separado.

## Datos de Power BI

| Archivo | Granularidad y tamaño |
|---|---|
| historico.csv | Fecha + tienda + familia; últimos 90 días, 160.380 filas |
| predicciones.csv | Fecha + tienda + familia + modelo; 85.536 filas, tres modelos |
| validacion.csv | Misma clave, test interno; 85.536 filas con sales, prediction, absolute_error y squared_error |
| tiendas.csv | Una fila por tienda; 54 filas |
| familias.csv | Una fila por familia; 33 filas |
| modelos.csv | Baseline semanal, Ridge y LightGBM; indicador del elegido |
| calendario.csv | Una fila por fecha desde el inicio del extracto hasta el final del forecast |

`horizon_day` identifica el paso 1–16. `cutoff` identifica cuándo se genera la previsión. En los CSV el alias `lightgbm` corresponde a `lightgbm_63` en esta ejecución; la correspondencia se guarda en `run_metadata.json`. Las relaciones del modelo semántico son de muchos a uno y de dirección única hacia las tablas de hechos. No se relacionan entre sí las tablas de hechos.

MAE = media del error absoluto. RMSE = raíz de la media del error cuadrático. WAPE = suma de errores absolutos / suma de ventas; queda indefinido si el denominador es cero. Las métricas de Power BI se calculan sobre las filas filtradas, no como promedio de porcentajes por segmento.
