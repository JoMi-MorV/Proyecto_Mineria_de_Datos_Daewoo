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
│       ├── __init__.py                    <-- Inicializador del paquete Python
│       ├── ingesta.py                     <-- Carga y validación inicial del dataset
│       ├── estadisticos.py                <-- Métricas descriptivas (media, skew, kurt, IQR)
│       ├── graficos.py                    <-- Generación modular de Histogramas y Boxplots
│       └── categoricas.py                 <-- Frecuencias absolutas y proporciones de categoríales
│
├── README.md
└── requirements.txt