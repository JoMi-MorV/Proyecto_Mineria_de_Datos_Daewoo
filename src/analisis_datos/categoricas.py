import pandas as pd

def calcular_frecuencias_categoricas(df, columnas=None):
    """
    Calcula la frecuencia absoluta y relativa para variables categóricas.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset con la información de Daewoo Steel Co.
    columnas : list, optional
        Lista de nombres de columnas categóricas a analizar. 
        Si es None, analiza las columnas por defecto ("WeekStatus", "DayOfWeek", "Load_Type").
        
    Returns:
    --------
    dict : Diccionario donde cada clave es el nombre de la variable 
           y el valor es un DataFrame con su tabla de frecuencias.
    """
    if columnas is None:
        columnas = ["WeekStatus", "Day_of_week", "Load_Type"]
        
    resultado_tablas = {}
    
    # Orden lógico para los días de la semana si están en inglés o español
    dias_orden = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"
    ]
    
    for col in columnas:
        if col in df.columns:
            # Obtener frecuencias
            if col == "Day_of_week":
                # Filtrar orden según los días presentes en el dataset
                orden_existente = [d for d in dias_orden if d in df[col].unique()]
                if orden_existente:
                    abs_freq = df[col].value_counts().reindex(orden_existente).dropna()
                else:
                    abs_freq = df[col].value_counts()
            else:
                abs_freq = df[col].value_counts()
            
            # Frecuencia Relativa (%)
            rel_freq = (abs_freq / len(df) * 100).round(2)
            
            # Construir DataFrame final
            df_freq = pd.DataFrame({
                "Frecuencia Absoluta (N°)": abs_freq.astype(int),
                "Proporción Relativa (%)": rel_freq.map("{:.2f}%".format)
            })
            
            df_freq.index.name = col
            resultado_tablas[col] = df_freq
            
    return resultado_tablas