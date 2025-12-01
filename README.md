# Valora — Predicción de desempeño empresarial por sector

Resumen ejecutivo
-----------------
Valora es una herramienta desarrollada para fortalecer la capacidad institucional de estimar el crecimiento y las ganancias esperadas de las empresas en Colombia por sector económico. A partir de información histórica de las 10.000 empresas más grandes del país, Valora integra análisis descriptivo y modelos predictivos en un dashboard interactivo que facilita la toma de decisiones estratégicas en inversión, política pública y desarrollo empresarial.

Problema que aborda
-------------------
Las áreas misionales trabajan con cortes anuales y tableros descriptivos que muestran “qué pasó”, pero carecen de mecanismos predictivos que anticipen “qué pasará” — por ejemplo, la evolución del crecimiento sectorial y su dispersión entre empresas. Valora cubre esa necesidad al proveer proyecciones de ganancias empresariales y resúmenes sectoriales basados en modelos de machine learning y análisis histórico.

Objetivo general
-----------------
Desarrollar un modelo predictivo capaz de estimar las ganancias proyectadas de las empresas, integrando técnicas de analítica predictiva y descriptiva de datos en un dashboard interactivo que facilite la toma de decisiones estratégicas.

Impacto esperado
----------------
- Mejorar la comprensión de las dinámicas económicas empresariales en Colombia.
- Proporcionar un ejemplo replicable de uso de datos abiertos con inteligencia artificial.
- Facilitar decisiones estratégicas en inversión, política pública o desarrollo empresarial.

Metodología
--------------------
1. Ingesta y limpieza: normalización de columnas monetarias, tratamiento de nulos y conversión de tipos.
2. Ingeniería de características: variables financieras y categóricas armonizadas por macrosector y región.
3. Modelado: comparación entre modelos (p. ej. Random Forest, XGBoost) para predecir ganancias; selección con validación cruzada y métricas (R², MAE, RMSE).
4. Validación y calibración: pruebas en conjuntos temporales y evaluación de sesgos sectoriales.
5. Despliegue en dashboard: inferencia en línea y visualizaciones interactivas.

Métricas de evaluación recomendadas
---------------------------------
- R² (explicación de la varianza)
- MAE (error absoluto medio)
- RMSE (raíz del error cuadrático medio)
- Evaluación por sector y por percentiles para medir dispersión y equidad de las predicciones

Autores
----------------
- Robert Josué Hernández Bautista
- Juan José Cifuentes Rodríguez
- Leidy Daniela Alzate Florez