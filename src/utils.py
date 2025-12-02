import numpy as np
import pandas as pd

def procesar_datos_usuario(inputs):
    """
    Convierte los inputs del usuario en un DataFrame listo para el modelo.
    VERSION BLINDADA: Maneja divisiones por cero y logaritmos negativos.
    """
    # 1. Convertir a DataFrame
    df = pd.DataFrame([inputs])
    
    # 2. Ingeniería de Variables (Ratios)
    
    # PROTECCIÓN CLAVE: Si Activos es 0, lo volvemos NaN para que no de error (Infinito)
    # El modelo tiene un Imputer que rellenará este NaN después.
    activos_safe = df['TOTAL ACTIVOS'].replace(0, np.nan)
    ingresos_safe = df['INGRESOS OPERACIONALES'].replace(0, np.nan)
    
    # Ratios Financieros
    df['RATIO_ENDEUDAMIENTO'] = df['TOTAL PASIVOS'] / activos_safe
    df['RATIO_PATRIMONIAL'] = df['TOTAL PATRIMONIO'] / activos_safe
    
    # ROA Anterior (Eficiencia reciente)
    df['ROA_ANTERIOR'] = df['GANANCIA_ANTERIOR'] / activos_safe
    
    # Margen (Solo informativo, si lo usas)
    df['MARGEN_ANTERIOR'] = df['GANANCIA_ANTERIOR'] / ingresos_safe

    # 3. Transformaciones Logarítmicas (Log1p)
    cols_a_log = ['TOTAL ACTIVOS', 'INGRESOS OPERACIONALES', 'TOTAL PASIVOS', 'TOTAL PATRIMONIO']
    
    for col in cols_a_log:
        # PROTECCIÓN CLAVE: np.maximum(x, 0) evita log de negativos
        valor_seguro = np.maximum(df[col], 0)
        df[f'LOG_{col}'] = np.log1p(valor_seguro)
        
    # 4. LIMPIEZA FINAL DE INFINITOS
    # Si algo se escapó y generó inf, lo matamos aquí
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
    return df

def predecir_ganancia_futura(modelo, df_procesado, ganancia_actual):
    """
    Calcula el valor final sumando el delta predicho.
    """
    try:
        # El modelo predice el CAMBIO (Delta)
        delta_predicho = modelo.predict(df_procesado)[0]
        
        # Sumamos al valor actual
        ganancia_futura = ganancia_actual + delta_predicho
        
        return ganancia_futura, delta_predicho
    except Exception as e:
        # Si falla, devolvemos el error para verlo en pantalla
        raise e