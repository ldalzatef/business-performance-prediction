import numpy as np
import pandas as pd

def procesar_datos_usuario(inputs):
    """
    Convierte los inputs del usuario (diccionario) en un DataFrame
    con las mismas transformaciones (Logs, Ratios) usadas en el entrenamiento.
    """
    # 1. Convertir a DataFrame
    df = pd.DataFrame([inputs])
    
    # 2. Ingeniería de Variables (Ratios)
    # Evitamos división por cero sumando un valor minúsculo si hace falta, 
    # aunque validaremos en el frontend.
    
    # Endeudamiento: Qué tanto de mis activos debo
    df['RATIO_ENDEUDAMIENTO'] = df['TOTAL PASIVOS'] / df['TOTAL ACTIVOS']
    
    # Solidez: Qué tanto es mío realmente
    df['RATIO_PATRIMONIAL'] = df['TOTAL PATRIMONIO'] / df['TOTAL ACTIVOS']
    
    # ROA Anterior: Qué tan eficiente fui el año pasado
    # (Ganancia Anterior / Activos Actuales) -> Proxy de eficiencia reciente
    df['ROA_ANTERIOR'] = df['GANANCIA_ANTERIOR'] / df['TOTAL ACTIVOS']
    
    # Margen: Cuánto me queda de cada venta
    if df['INGRESOS OPERACIONALES'].iloc[0] != 0:
        df['MARGEN_ANTERIOR'] = df['GANANCIA_ANTERIOR'] / df['INGRESOS OPERACIONALES']
    else:
        df['MARGEN_ANTERIOR'] = 0

    # 3. Transformaciones Logarítmicas (Log1p)
    # Aplicamos logaritmo a las variables de volumen para suavizar escalas
    # OJO: Solo a las variables de tamaño (siempre positivas)
    cols_a_log = ['TOTAL ACTIVOS', 'INGRESOS OPERACIONALES', 'TOTAL PASIVOS', 'TOTAL PATRIMONIO']
    
    for col in cols_a_log:
        # PROTECCIÓN: Aseguramos que no haya negativos antes del log
        # np.maximum compara el valor con 0 y se queda con el mayor
        valor_seguro = np.maximum(df[col], 0)
        df[f'LOG_{col}'] = np.log1p(valor_seguro)

def predecir_ganancia_futura(modelo, df_procesado, ganancia_actual):
    """
    Calcula el valor final.
    Ecuación: Futuro = Actual + Delta_Predicho
    """
    # 1. El modelo predice el CAMBIO (Delta)
    delta_predicho = modelo.predict(df_procesado)[0]
    
    # 2. Sumamos al valor actual
    ganancia_futura = ganancia_actual + delta_predicho
    
    return ganancia_futura, delta_predicho