# Informe nativo de Power BI

## Abrir el proyecto

1. Clona o descarga el repositorio completo. Instala una versión reciente de **Power BI Desktop para Windows** compatible con proyectos PBIP e informes PBIR. Si la versión los requiere como funciones de vista previa, activa esas funciones y reinicia Desktop.
2. Abre `Favorita.pbip` desde esta carpeta. Conserva a su lado `Favorita.Report`, `Favorita.SemanticModel` y `data`.
3. Pulsa **Actualizar**. El parámetro `DataBaseURL` apunta a los CSV de la rama `main` de este repositorio. Para la fuente web pública selecciona autenticación **Anónima**; el nivel de privacidad aplicable a esos datos públicos es **Público**.
4. Para trabajar sin acceso a GitHub, en **Transformar datos → Editar parámetros** pon `UseLocalFiles = true` y cambia `LocalDataFolder` por la ruta absoluta de tu carpeta `powerbi/data`, por ejemplo `C:/TFM/prediccion-demanda-favorita/powerbi/data`. Aplica y actualiza. El directorio no debe terminar con `/`.
5. Comprueba las dos páginas, cambia tienda, familia y modelo, y confirma que las tablas y los gráficos responden. El modelo utilizado por defecto es LightGBM. Si quieres entregar también un `.pbix`, guárdalo desde Desktop una vez actualizado.

## Páginas y recorrido

**Previsión:** selecciona tienda y familia para revisar las ventas recientes y las previsiones del 16–31/08/2017. El selector de modelo permite comparar baseline, Ridge y LightGBM. Los totales de varias familias sirven como agregados del campo `sales`, pero pueden mezclar unidades físicas. La tabla de previsiones contiene el detalle por día. Desde sus tres puntos usa **Exportar datos**; el informe habilita exportación resumida.

**Evaluación:** consulta las métricas del test interno del 31/07 al 15/08/2017 y el error por familia y horizonte. El selector de modelo afecta a esas métricas. Los filtros de tienda y familia también se mantienen; las medidas de calidad quitan el filtro de calendario para evitar que una selección del forecast final deje el test interno vacío. Por eso sus tarjetas indican expresamente el periodo de evaluación.

El usuario puede consultar y exportar una previsión para revisarla con el responsable de tienda. No se generan pedidos ni decisiones de stock.

## Modelo y datos

El modelo incluye calendario, tiendas, familias, modelos y tres tablas de hechos: histórico, previsión y validación. Son 8 tablas contando la tabla de medidas, 11 relaciones y 17 medidas DAX. Las tres tablas de hechos tienen relaciones de dirección única con sus dimensiones. El CSV de validación contiene 28.512 filas por modelo; las ventas reales solo se utilizan para medir el error de ese bloque.

El formato PBIP permite revisar los cambios del modelo y del informe en Git. `model.bim` contiene el modelo semántico, las consultas Power Query y las medidas. `Favorita.Report/definition` contiene las páginas y las 28 definiciones de visuales. No se deben subir las cachés locales `.pbi`.

## Verificación realizada y pendiente

Se han validado los JSON contra los esquemas oficiales de Microsoft, las referencias a columnas y medidas, las claves, las relaciones, el número de predicciones y la igualdad de MAE, RMSE y WAPE con los resultados de Python. Evidencia: [powerbi_validation.json](../outputs/powerbi_validation.json).

**El entorno de ejecución no dispone de Power BI Desktop. La apertura, el refresco, las medidas DAX en Desktop y la interacción visual deben verificarse en Windows antes de la defensa.** No se presenta una captura de Desktop ni se afirma que se haya publicado en Power BI Service.

Para repetir la comprobación estructural:

```bash
python scripts/build_powerbi.py
git clone --depth 1 https://github.com/microsoft/json-schemas.git /ruta/json-schemas
python scripts/validate_powerbi.py --schemas /ruta/json-schemas
```

Sin `--schemas` el validador comprueba datos y referencias, pero no declara haber validado los esquemas. [Documentación oficial de proyectos PBIP](https://learn.microsoft.com/en-us/power-bi/developer/projects/projects-overview).

## Apertura observada en el equipo del alumno

El 04/10/2026 el alumno abrió el proyecto y aportó una captura de Power BI Desktop con las dos páginas y los datos cargados. En la página de previsión se observan MAE 75,85, WAPE 16,2% y mejora de MAE 21,4%, coherentes con los resultados de Python. La captura confirma la apertura y la visualización inicial; aún no confirma un refresco explícito, todos los filtros ni la exportación. Se han corregido los textos recortados de las tarjetas. Registro: [desktop_review.json](../outputs/desktop_review.json). Para la entrega cómoda al profesor, guardar un PBIX desde Desktop después de verificar la actualización.
