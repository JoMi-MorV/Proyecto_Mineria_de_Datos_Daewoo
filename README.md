# Caso Daewoo para el ramo de minería de datos
En este proyecto se realiza un caso de la industria Daewoo aplicando conceptos y teoría vista en el ramo de minería de datos

# Estructura de archivos

daewoo_steel_project/
│
├── data/
│   └── raw/
│       └── Steel_industry_data.csv        <-- Dataset original
│
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb     <-- Tu reporte en Jupyter
│
├── src/
│   └── analisis_datos/
│       ├── __init__.py                    <-- Módulo Python
│       ├── ingesta.py                     <-- Módulo para cargar/gestionar el dataset
│       └── estadisticos.py                <-- Módulo de cálculos y gráficos
│
├── README.md
└── requirements.txt