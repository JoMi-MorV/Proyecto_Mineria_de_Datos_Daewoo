# Caso Daewoo para el ramo de minería de datos

Este proyecto analiza datos de consumo energético de la industria del acero mediante estadística descriptiva, visualizaciones y revisión de valores atípicos. El análisis se desarrolla en un notebook y utiliza módulos Python reutilizables para cargar y procesar los datos.

# Estructura de archivos

Proyecto_Mineria_de_Datos_Daewoo/
│
├── data/
│   ├── raw/
│   │   └── Steel_industry_data.csv        # Dataset original
│   └── clean/
│       └── Steel_industry_clean.csv       # Dataset exportado tras el tratamiento
│
├── notebooks/
│   └── Tarea_2.ipynb     # Análisis exploratorio y ejecución del flujo
│
├── src/
│   └── analisis_datos/
│       ├── __init__.py                    # Inicializador del paquete
│       ├── ingesta.py                     # Localiza la raíz y carga el CSV original
│       ├── estadistico.py                 # Estadísticos descriptivos y agrupación por rangos
│       ├── categoricas.py                 # Frecuencias de variables categóricas
│       ├── graficos.py                    # Histogramas, KDE y gráficos de variables categóricas
│       └── outliers.py                    # Nulos, IQR, boxplots, auditorías e imputación
│
├── README.md
└── requirements.txt

El notebook parte del archivo original en `data/raw/` y guarda el dataset tratado en `data/clean/Steel_industry_clean.csv`. Las carpetas del entorno virtual (`venv/`) y los checkpoints de Jupyter son locales y no forman parte del código fuente del proyecto.