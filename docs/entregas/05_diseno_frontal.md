# Entrega 5 - Diseño del frontal y experiencia de usuario

**Alumno:** Adel Toutouh El Bouchti  
**Proyecto:** Data Science / IA  
**Proyecto seleccionado:** Predicción de ventas para Corporación Favorita

## 1. Resumen de la solución y del usuario

El proyecto busca predecir las ventas de los siguientes 16 días para cada combinación de tienda y familia de producto. Como ya se indicó en la entrega anterior, el modelo predice **ventas registradas** y no demanda real, porque el dataset no contiene información de inventario ni roturas de stock.

El usuario principal que tengo en mente es un **analista o una persona responsable de planificación** que quiera revisar una previsión antes de tomar una decisión. No se pretende que el sistema compre inventario automáticamente, sino que sirva como una herramienta de apoyo para consultar previsiones, compararlas con el comportamiento histórico y detectar dónde el modelo tiene más error.

El frontal será un **dashboard en Power BI**. La pantalla principal permitirá elegir una tienda y una familia de producto, ver la previsión de los siguientes 16 días y consultar métricas de validación del modelo.

La acción principal será revisar la previsión y, si el usuario la necesita para trabajar con ella fuera del dashboard, poder **exportar el detalle a CSV**.

## 2. Imagen mockup del frontal

El siguiente mockup representa la pantalla principal que quiero construir para el MVP.

![Mockup del frontal](../assets/05_mockup_frontal.png)

Los números que aparecen en el mockup son únicamente valores de ejemplo para representar el diseño. No son resultados reales del modelo todavía.

La pantalla está dividida en cuatro zonas principales:

- filtros de tienda, familia y modelo;
- indicadores principales de la previsión y del error;
- gráfico con histórico reciente y los 16 días previstos;
- detalle de la previsión con opción de exportación.

También he incluido un aviso visible para recordar que la salida representa ventas previstas y no una recomendación automática de compra.

## 3. Justificación del diseño

### 3.1. Utilidad y valor de la solución

La idea del frontal es que el usuario no tenga que revisar directamente ficheros CSV o resultados del modelo en Python. Desde una sola pantalla puede seleccionar la parte del negocio que quiere consultar y ver la previsión con algo de contexto.

La información que considero más importante es:

- ventas previstas para el horizonte seleccionado;
- evolución histórica reciente;
- MAE obtenido en validación;
- comparación del modelo con el baseline;
- promociones conocidas para las fechas futuras;
- detalle diario de las predicciones.

He decidido no mostrar métricas más técnicas del entrenamiento, hiperparámetros o información interna del pipeline en la pantalla principal porque no ayudan a la tarea diaria del usuario. Si hicieran falta, podrían aparecer más adelante en una página de detalle del modelo.

El dashboard tampoco convertirá la predicción en una orden de compra. La utilidad está en ofrecer una referencia para que el usuario pueda revisar la planificación con más información, manteniendo la decisión final fuera del modelo.

### 3.2. Flujo de usuario

El recorrido principal sería el siguiente:

1. El usuario abre el dashboard y ve una vista general con una tienda y una familia seleccionadas.
2. Cambia los filtros de **tienda** y **familia de producto** según lo que quiera analizar.
3. El dashboard carga las predicciones generadas por el modelo para los siguientes 16 días.
4. Primero puede revisar los KPI: ventas previstas, MAE de validación y mejora frente al baseline.
5. Después consulta el gráfico para ver cómo continúa la previsión respecto al histórico reciente.
6. Si necesita más detalle, revisa la tabla diaria y las promociones previstas.
7. Finalmente puede exportar los datos a CSV para analizarlos o utilizarlos en otro informe.

El modelo no se entrenará cada vez que el usuario cambie un filtro. Las predicciones estarán calculadas previamente por el pipeline y Power BI se encargará de filtrar y mostrar los resultados correspondientes.

#### Situaciones excepcionales

También quiero contemplar algunos casos sencillos:

- Si no existen predicciones para una combinación de tienda y familia, se mostrará un mensaje indicando que no hay datos disponibles.
- Si falta alguna variable necesaria para generar la predicción, esa combinación no se presentará como si el resultado fuera válido.
- Si el error de validación de una familia o tienda es especialmente alto, se podrá mostrar una advertencia para que el usuario interprete la previsión con más cuidado.
- Mientras se actualizan los datos, Power BI mostrará su indicador de carga habitual en lugar de dejar la pantalla sin respuesta.

### 3.3. Experiencia de usuario

He intentado que la pantalla sea sencilla porque el objetivo no es enseñar todas las variables del proyecto, sino permitir que una persona encuentre rápidamente una previsión concreta.

**Jerarquía visual.** En la parte superior aparecen los KPI principales. Después ocupa más espacio el gráfico de histórico y previsión, porque es la parte que mejor permite entender el resultado. Los filtros quedan en el lateral para que estén disponibles sin ocupar la zona central.

**Simplicidad.** La pantalla principal tendrá pocos filtros: tienda, familia y, si finalmente comparo más de un modelo en el dashboard, el modelo. El horizonte del MVP será de 16 días y no quiero añadir muchas opciones que luego no tengan una utilidad clara.

**Legibilidad y consistencia.** Se utilizarán los mismos nombres que en el resto del proyecto, por ejemplo `store_nbr`, familia, ventas previstas y MAE, pero en el frontal se mostrarán con etiquetas entendibles. Los valores importantes tendrán también texto y no dependerán solo del color.

**Contexto y confianza.** La previsión aparecerá junto al histórico, al error de validación y a la mejora frente al baseline. También habrá un aviso explicando que una predicción no es una cifra segura y que el proyecto no conoce el inventario real.

**Control del usuario.** El usuario podrá cambiar los filtros, comparar distintas combinaciones y decidir si exporta o no el resultado. El sistema no ejecutará decisiones automáticas.

**Accesibilidad.** El MVP estará pensado principalmente para ordenador, porque Power BI se utilizará como herramienta de análisis. Evitaré utilizar letras pequeñas y procuraré que los contrastes sean suficientes. Si se usan colores para diferenciar histórico y previsión, también se distinguirán mediante etiquetas y tipo de línea.

## 4. Presentación de resultados y explicabilidad

El resultado principal será la **previsión diaria de ventas para los siguientes 16 días**.

No quiero mostrar únicamente una cifra total. Para interpretar el resultado se enseñarán también:

- el histórico reciente de ventas;
- el inicio exacto del horizonte de predicción;
- el MAE obtenido en validación temporal;
- la mejora o empeoramiento frente al baseline;
- promociones futuras conocidas;
- una tabla con la predicción de cada día.

Además, si la validación muestra que el error aumenta en los últimos días por el uso de predicción recursiva, esa limitación se indicará en el dashboard o en la documentación. De esta manera no se presentarán los 16 valores como si tuvieran el mismo nivel de fiabilidad.

La información más técnica, como los hiperparámetros del modelo, transformaciones internas o detalles completos del backtesting, no estará en la pantalla principal. Esa información pertenece más a la parte de desarrollo y evaluación del modelo que al uso normal del frontal.

### IA generativa

En el MVP **no voy a utilizar IA generativa como capa de explicación**. En este proyecto el valor principal está en mostrar de forma clara las predicciones, el histórico y las métricas de error. Añadir un modelo generativo para redactar explicaciones aumentaría la complejidad y podría introducir interpretaciones que no estén directamente justificadas por los datos.

## 5. Alcance del MVP

La parte que quiero tener realmente implementada al final del curso es un dashboard de Power BI conectado a los resultados generados por el pipeline de Python.

El MVP debería permitir:

- filtrar por tienda y familia de producto;
- consultar las previsiones de 16 días;
- visualizar histórico y previsión en un gráfico;
- mostrar MAE y comparación con el baseline;
- consultar el detalle diario de la previsión;
- mostrar promociones conocidas cuando estén disponibles;
- exportar los datos mostrados.

El mockup incluye algunos elementos de presentación que podrán ajustarse cuando existan resultados reales, como los valores concretos de KPI, mensajes de estado o la disposición final de algunos bloques.

La tecnología prevista es:

- **Python** para preparación de datos, generación de variables, entrenamiento y predicción;
- **Parquet o CSV** para guardar los resultados que consumirá el frontal;
- **Power BI** para construir el dashboard.

No voy a desarrollar una aplicación web independiente ni un sistema que actualice las predicciones en tiempo real. Para el alcance del curso me parece más realista generar los resultados con el pipeline y consumirlos después desde Power BI.
