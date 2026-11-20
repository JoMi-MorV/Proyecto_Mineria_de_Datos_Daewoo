# Caso Daewoo para el ramo de minería de datos

Este proyecto analiza datos de consumo energético de la industria del acero mediante estadística descriptiva, visualizaciones y revisión de valores atípicos. El análisis se desarrolla en varios notebooks y utiliza módulos Python reutilizables para cargar y procesar los datos.

# Estructura de archivos

```text
Proyecto_Mineria_de_Datos_Daewoo/
├── data/
│   ├── raw/
│   │   └── Steel_industry_data.csv          # Dataset original
│   └── clean/
│       └── Steel_industry_clean.csv         # Dataset procesado para el análisis
├── notebooks/
│   ├── Tarea_2.ipynb                        # Análisis exploratorio y preparación de datos
│   ├── Tarea_3_MD.ipynb                     # Aplicación de clustering
│   ├── Tarea_4_MD.ipynb                     # Árboles de decisión, Random Forest y AdaBoost
│   └── Tarea_5_MD.ipynb                     # SVM y métodos de validación
├── src/
│   ├── analisis_datos/
│   │   ├── __init__.py
│   │   ├── categoricas.py                   # Frecuencias de variables categóricas
│   │   ├── estadistico.py                   # Estadísticos descriptivos
│   │   ├── graficos.py                      # Gráficos exploratorios
│   │   ├── ingesta.py                       # Localización de la raíz y carga de datos
│   │   └── outliers.py                      # Detección, auditoría e imputación de atípicos
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── jerarquico.py                    # Clustering jerárquico
│   │   ├── knn.py                            # Modelos KNN y métricas
│   │   └── preparacion.py                    # Preparación de datos para clustering
│   ├── modelos_aprendizaje/
│   │   ├── __init__.py
│   │   ├── adaboost.py                       # Modelo AdaBoost
│   │   ├── arbol_decision.py                 # Modelo de árbol de decisión
│   │   ├── evaluacion_modelo.py              # Evaluación de modelos
│   │   ├── preparacion.py                    # Preparación de datos
│   │   ├── pronostico.py                     # Funciones de pronóstico
│   │   ├── random_forest.py                  # Modelo Random Forest
│   │   └── redes_bayesianas.py               # Modelos de redes bayesianas
│   └── SVM_CV/
│       ├── __init__.py
│       ├── datos.py                          # Preparación y carga de datos
│       ├── diagnosticos.py                   # Diagnósticos de modelos
│       ├── graficos.py                       # Gráficos de análisis
│       ├── modelos_svm.py                    # Modelos SVM
│       └── validacion_cruzada.py             # Validación cruzada
│
├── .gitignore
├── README.md
└── requirements.txt
```

Los notebooks utilizan los datos de `data/`; el dataset procesado se encuentra en `data/clean/Steel_industry_clean.csv`. Las carpetas del entorno virtual (`venv/`) y los checkpoints de Jupyter son locales y no forman parte del código fuente del proyecto.