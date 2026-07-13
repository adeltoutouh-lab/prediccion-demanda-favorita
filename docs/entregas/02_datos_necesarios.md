# Entrega 2 - Selección de idea y análisis de datos necesarios

**Alumno:** Adel Toutouh El Bouchti  
**Proyecto:** Data Science / IA  
**Idea seleccionada:** Predicción de demanda para Corporación Favorita

## 1. Idea seleccionada

### Problema que resuelve

La idea seleccionada consiste en desarrollar un sistema de predicción de demanda aplicado a Corporación Favorita, una empresa real del sector de supermercados y retail. La necesidad que se quiere abordar es estimar las ventas futuras de cada familia de productos en cada tienda. Una previsión poco precisa puede provocar falta de existencias, pérdida de ventas o exceso de inventario. El problema es especialmente relevante en alimentación, donde parte de los productos son perecederos y la demanda puede cambiar por promociones, festivos, estacionalidad o diferencias entre localidades. El valor del proyecto estaría en ofrecer una previsión que sirva como apoyo a la planificación, sin afirmar que el modelo se encuentra implantado actualmente en la empresa.

### Solución planteada

La solución se desarrollará mediante un proceso de Data Science basado en datos históricos públicos de Corporación Favorita. Primero se realizará una limpieza y exploración de las ventas para identificar tendencias, estacionalidad, valores atípicos y diferencias entre tiendas y familias de producto. Después se crearán variables temporales, medias móviles y retardos de ventas, y se incorporará información sobre promociones, transacciones, festivos, características de las tiendas y precio del petróleo. Se comparará un modelo base sencillo con uno o varios modelos de machine learning, utilizando siempre una validación temporal para evitar utilizar información futura durante el entrenamiento.

### MVP del proyecto final

El producto mínimo viable estará formado por un pipeline reproducible en Python, un modelo capaz de generar previsiones para un horizonte de 16 días y un dashboard en Power BI. En el dashboard se podrá seleccionar una tienda y una familia de producto, consultar la evolución histórica, comparar valores reales y previstos, analizar el efecto de promociones y festivos y visualizar métricas de error. El MVP no intentará automatizar la compra de inventario ni calcular un ahorro real para la empresa, porque el dataset no incluye costes, niveles de stock ni márgenes. Su función será demostrar que se puede construir una herramienta de apoyo a la planificación con datos reales y públicos.

## 2. Datos necesarios

### 2.1 Unidad de análisis y granularidad

La unidad principal de análisis será una combinación de:

- fecha;
- tienda;
- familia de producto.

Por tanto, la granularidad será **diaria por tienda y familia de producto**. Esta granularidad permite estudiar la demanda con suficiente detalle sin llegar al nivel de cliente individual. También se utilizarán tablas auxiliares con información diaria por tienda, diaria a nivel nacional y datos estáticos de cada establecimiento.

### 2.2 Variable objetivo

La variable objetivo será:

- `sales`: ventas registradas para una familia de productos, en una tienda y una fecha determinada.

El objetivo del modelo será estimar esta variable para fechas futuras.

### 2.3 Variables necesarias

| Grupo | Variables o campos | Uso previsto |
|---|---|---|
| Identificación temporal | `date` | Ordenar la serie y crear variables de calendario |
| Tienda | `store_nbr` | Diferenciar el comportamiento de cada establecimiento |
| Producto | `family` | Diferenciar las familias de productos |
| Objetivo | `sales` | Entrenamiento y evaluación del modelo |
| Promociones | `onpromotion` | Medir el posible efecto de productos promocionados |
| Características de tienda | `city`, `state`, `type`, `cluster` | Comparar tiendas según localización y tipología |
| Actividad de tienda | `transactions` | Aproximar el nivel diario de actividad |
| Festivos y eventos | `type`, `locale`, `locale_name`, `description`, `transferred` | Incorporar efectos de calendario |
| Contexto económico | `dcoilwtico` | Analizar el precio diario del petróleo como variable externa |

A partir de estas columnas se crearán variables derivadas:

- día de la semana, mes, trimestre y fin de semana;
- principio y final de mes;
- retardos de ventas de 1, 7, 14 y 28 días;
- medias móviles de 7, 14 y 28 días;
- ventas del mismo día de la semana anterior;
- días anteriores y posteriores a un festivo;
- intensidad promocional;
- tendencia temporal;
- medias históricas por tienda y familia.

Las variables derivadas se calcularán utilizando únicamente datos anteriores a la fecha que se desea predecir, para evitar fuga de información.

### 2.4 Profundidad histórica

La fuente principal contiene datos diarios desde 2013 hasta agosto de 2017 y un periodo de prueba de 16 días. Esto proporciona más de cuatro años de histórico, suficiente para analizar:

- patrones semanales;
- estacionalidad anual;
- festivos;
- promociones;
- diferencias entre tiendas;
- cambios de tendencia.

Aunque el histórico no es reciente, sí es adecuado para un proyecto académico centrado en la metodología de predicción. Esta limitación deberá explicarse y no se presentará el modelo como una herramienta lista para operar con la situación comercial actual de la empresa.

### 2.5 Volumen aproximado

El conjunto de entrenamiento tiene alrededor de tres millones de filas y combina aproximadamente 54 tiendas con 33 familias de producto. Es un volumen suficientemente grande para que el proyecto tenga sentido y para comparar modelos, pero todavía puede trabajarse en un ordenador personal utilizando tipos de datos optimizados, lectura por bloques o formatos como Parquet.

Para el primer prototipo se podrá trabajar con una selección de tiendas o familias. Una vez comprobado el pipeline, se ampliará el entrenamiento al conjunto completo si los recursos disponibles lo permiten.

### 2.6 Datos imprescindibles

Se consideran imprescindibles:

- fecha;
- identificador de tienda;
- familia de producto;
- ventas;
- número de productos en promoción;
- características básicas de las tiendas;
- calendario de festivos y eventos.

Sin la fecha, la tienda, la familia y las ventas no sería posible formular el problema de predicción. Las promociones y los festivos son necesarios para que el modelo no dependa únicamente del histórico de ventas.

### 2.7 Datos deseables, pero no obligatorios

Sería útil disponer también de:

- inventario disponible;
- roturas de stock reales;
- precios de venta;
- costes de compra y almacenamiento;
- margen por producto;
- caducidad;
- datos meteorológicos;
- acciones de la competencia;
- información a nivel de producto individual;
- pedidos realizados a proveedores.

Estos datos permitirían transformar la predicción en una recomendación de inventario y calcular impacto económico. Sin embargo, no están incluidos en la fuente pública principal y no se solicitarán directamente a la empresa. Por este motivo, el alcance se limitará a predecir ventas y analizar el error de previsión.

## 3. Fuentes de datos previstas

### 3.1 Fuente principal

**Store Sales - Time Series Forecasting, Kaggle**

- Página del proyecto:  
  https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- Página de descarga:  
  https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
- Empresa asociada a los datos: Corporación Favorita.
- Web corporativa:  
  https://www.corporacionfavorita.com/

La fuente es pública y no requiere contactar con la empresa ni pagar por los datos. Sin embargo, para descargarla es necesario crear una cuenta gratuita de Kaggle y aceptar las condiciones de la competición. Por tanto, es accesible para un proyecto académico, pero su uso sigue estando sujeto a los términos indicados por Kaggle.

### 3.2 Archivos esperados

| Archivo | Contenido principal | Formato |
|---|---|---|
| `train.csv` | Histórico de ventas y promociones | CSV |
| `test.csv` | Fechas y combinaciones que deben predecirse | CSV |
| `stores.csv` | Ciudad, provincia, tipo y clúster de tienda | CSV |
| `transactions.csv` | Número diario de transacciones por tienda | CSV |
| `holidays_events.csv` | Festivos y eventos nacionales, regionales y locales | CSV |
| `oil.csv` | Precio diario del petróleo | CSV |
| `sample_submission.csv` | Ejemplo del formato de predicción | CSV |

Existe un histórico completo dentro del periodo publicado. La fuente es estable porque la competición y sus archivos permanecen alojados en Kaggle, aunque no se trata de una API en tiempo real ni de un dataset que se actualice periódicamente.

### 3.3 Calidad y riesgos detectados

Los principales riesgos son:

1. **Antigüedad de los datos.** El histórico termina en 2017, por lo que el proyecto demostrará una metodología y no representará necesariamente el comportamiento actual de la empresa.
2. **Ausencia de inventario y precios.** No se puede distinguir con seguridad entre una demanda baja y una venta baja causada por falta de stock. Tampoco se puede calcular el beneficio o ahorro económico real.
3. **Datos agregados.** Las ventas están agrupadas por familia de producto y no por SKU individual.
4. **Valores ausentes.** Algunas variables auxiliares, como el precio del petróleo, pueden requerir imputación.
5. **Eventos excepcionales.** Existen acontecimientos que pueden alterar temporalmente las ventas y que deberán estudiarse para decidir si se modelan como eventos.
6. **Acceso mediante Kaggle.** La descarga depende de una cuenta y de la aceptación de sus condiciones.
7. **Coste computacional.** La creación de retardos y medias móviles sobre millones de registros puede consumir memoria si no se optimiza el proceso.
8. **Fuga temporal.** Una división aleatoria de entrenamiento y prueba produciría resultados poco fiables. La validación deberá respetar el orden cronológico.

### 3.4 Alternativa si falla la fuente principal

Si no fuera posible acceder al dataset de Corporación Favorita, la alternativa sería **Rossmann Store Sales**, también disponible en Kaggle:

https://www.kaggle.com/competitions/rossmann-store-sales

Esta fuente plantea un problema similar de predicción de ventas por tienda y día. Permitiría mantener casi el mismo objetivo, las mismas fases del proyecto y un MVP parecido, aunque con menos información sobre familias de productos.

Como segunda medida de reducción de riesgo, si el conjunto principal resultara demasiado pesado se mantendría Corporación Favorita, pero se limitaría el alcance a una selección representativa de tiendas y familias. De esta forma no sería necesario cambiar de problema ni de empresa.

## 4. Consideraciones de privacidad y protección de datos

El dataset no incluye nombres de clientes, correos electrónicos, teléfonos, documentos de identidad ni otra información personal identificable. Las observaciones corresponden a ventas agregadas por fecha, tienda y familia de producto. Tampoco se incluyen perfiles individuales de empleados o proveedores.

Por este motivo, no será necesario anonimizar personas. Aun así, se aplicarán las siguientes medidas:

- no intentar identificar clientes ni reconstruir comportamientos individuales;
- utilizar los datos únicamente con fines académicos;
- respetar las condiciones de uso de Kaggle;
- citar la procedencia del dataset;
- no publicar archivos si las condiciones de la competición impiden redistribuirlos;
- mantener los CSV fuera del repositorio mediante `.gitignore`;
- publicar únicamente código, documentación, resultados agregados y, cuando sea necesario, pequeñas muestras permitidas.

Desde el punto de vista ético, el modelo no tomará decisiones sobre personas ni utilizará variables sensibles. El principal riesgo sería presentar las predicciones como decisiones exactas. Para evitarlo, el dashboard mostrará métricas de error y se describirá el sistema como una herramienta de apoyo, no como un sustituto de la decisión profesional.

## 5. Viabilidad inicial del proyecto

### 5.1 Obtención de los datos

La obtención parece viable porque los archivos están publicados en Kaggle y no dependen de la autorización directa de Corporación Favorita. El único requisito relevante es disponer de una cuenta gratuita y aceptar las condiciones de uso.

### 5.2 Suficiencia de calidad, granularidad e histórico

La granularidad diaria por tienda y familia de producto es adecuada para un problema de predicción de demanda. El histórico de más de cuatro años permite estudiar patrones semanales y anuales. Además, las tablas auxiliares aportan promociones, festivos, transacciones y características de las tiendas. La principal limitación es la falta de variables de inventario, precios y costes.

### 5.3 Desarrollo realista durante el curso

El proyecto es realista si se desarrolla por fases:

1. carga, limpieza y análisis exploratorio;
2. creación de un modelo base;
3. ingeniería de variables;
4. entrenamiento de un modelo de machine learning;
5. validación temporal y comparación de resultados;
6. generación de predicciones;
7. dashboard final.

El MVP no dependerá de utilizar redes neuronales ni de construir una aplicación compleja. Un modelo base y un modelo de árboles de gradiente bien evaluados, acompañados de un dashboard funcional, serían suficientes para demostrar el valor del proyecto.

### 5.4 Parte más arriesgada

La parte con mayor riesgo es trabajar correctamente con muchas series temporales y evitar fugas de información al crear las variables. También puede ser costoso entrenar sobre los tres millones de registros. Para reducir este riesgo se empezará con una muestra, se usarán divisiones cronológicas y se conservará un modelo base sencillo durante todo el desarrollo.

### 5.5 Plan alternativo

El plan alternativo principal será reducir el número de tiendas o familias manteniendo la misma fuente. Si el acceso a Kaggle fallara por completo, se utilizaría Rossmann Store Sales. En ambos casos se conservaría el objetivo de predecir ventas diarias y el producto final seguiría siendo un modelo de forecasting acompañado de un dashboard.

## Referencias

- Corporación Favorita. Sitio web corporativo: https://www.corporacionfavorita.com/
- Kaggle. Store Sales - Time Series Forecasting: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- Kaggle. Store Sales - Time Series Forecasting, datos: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
- Kaggle. Rossmann Store Sales: https://www.kaggle.com/competitions/rossmann-store-sales
