# Caso Daewoo para el ramo de minería de datos

Este proyecto analiza datos de consumo energético de la industria del acero mediante estadística descriptiva, visualizaciones y revisión de valores atípicos. El análisis se desarrolla en un notebook y utiliza módulos Python reutilizables para cargar y procesar los datos.

# Estructura de archivos

Proyecto_Mineria_de_Datos_Daewoo/
├── data/
│   ├── raw/
│   │   └── Steel_industry_data.csv          # Dataset original
│   └── clean/
│       └── Steel_industry_clean.csv         # Dataset limpio para el análisis
├── notebooks/
│   ├── Tarea_2.ipynb                        # Análisis exploratorio y preparación de datos
│   └── Tarea_3.ipynb                        # Análisis de clustering 
├── src/
│   ├── analisis_datos/
│   │   ├── __init__.py
│   │   ├── categoricas.py                   # Frecuencias de variables categóricas
│   │   ├── estadistico.py                   # Estadísticos descriptivos
│   │   ├── graficos.py                      # Gráficos exploratorios
│   │   ├── ingesta.py                       # Localización de la raíz y carga de datos
│   │   └── outliers.py                      # Detección, auditoría e imputación de atípicos
│   └── clustering/
│       ├── __init__.py
│       ├── knn.py                            # Evaluación de modelos KNN y métricas
│       ├── preparacion.py                    # Preparación de datos para clasificación
│       └── jerarquico.py
│
├── .gitignore
├── README.md
└── requirements.txt

El notebook parte del archivo original en `data/raw/` y guarda el dataset tratado en `data/clean/Steel_industry_clean.csv`. Las carpetas del entorno virtual (`venv/`) y los checkpoints de Jupyter son locales y no forman parte del código fuente del proyecto.