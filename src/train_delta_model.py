import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def entrenar_modelo_delta():
    print("🚀 Iniciando entrenamiento del Modelo Delta (Crecimiento)...")
    
    # 1. Cargar Datos
    ruta_csv = 'data/greatest_colombian_business.csv'
    if not os.path.exists(ruta_csv):
        print("❌ Error: No se encuentra el archivo csv en data/")
        return

    df = pd.read_csv(ruta_csv)
    
    # 2. Limpieza de Moneda
    cols_num = ['INGRESOS OPERACIONALES', 'GANANCIA (PÉRDIDA)', 'TOTAL ACTIVOS', 'TOTAL PASIVOS', 'TOTAL PATRIMONIO']
    for col in cols_num:
        if df[col].dtype == 'O':
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. CREACIÓN DEL TARGET (DELTA)
    df = df.sort_values(['NIT', 'Año de Corte'])
    df['GANANCIA_ANTERIOR'] = df.groupby('NIT')['GANANCIA (PÉRDIDA)'].shift(1)
    df['DELTA_GANANCIA'] = df['GANANCIA (PÉRDIDA)'] - df['GANANCIA_ANTERIOR']
    df = df.dropna(subset=['GANANCIA_ANTERIOR', 'DELTA_GANANCIA'])
    
    print(f"📊 Datos listos: {len(df)} registros históricos.")

    # 4. Ingeniería de Variables (Feature Engineering)
    # --- Ratios (Protegidos contra división por cero) ---
    # Si Activos es 0, el ratio será NaN (luego el Imputer lo arregla), no Infinito.
    # Reemplazamos 0 por NaN temporalmente en el denominador
    activos_safe = df['TOTAL ACTIVOS'].replace(0, np.nan)
    
    df['RATIO_ENDEUDAMIENTO'] = df['TOTAL PASIVOS'] / activos_safe
    df['RATIO_PATRIMONIAL'] = df['TOTAL PATRIMONIO'] / activos_safe
    df['ROA_ANTERIOR'] = df['GANANCIA_ANTERIOR'] / activos_safe
    
    # --- Logs (Protegidos contra negativos e infinitos) ---
    for col in ['TOTAL ACTIVOS', 'INGRESOS OPERACIONALES', 'TOTAL PASIVOS', 'TOTAL PATRIMONIO']:
        # np.maximum(x, 0) evita logs de negativos
        df[f'LOG_{col}'] = np.log1p(np.maximum(df[col], 0))

    # --- LIMPIEZA FINAL DE INFINITOS (EL FIX CLAVE) ---
    # Reemplazamos cualquier inf generado por un NaN (que el SimpleImputer sí sabe manejar)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 5. Definir Features y Target
    features_numericas = [
        'LOG_TOTAL ACTIVOS', 'LOG_INGRESOS OPERACIONALES', 'LOG_TOTAL PASIVOS',
        'RATIO_ENDEUDAMIENTO', 'RATIO_PATRIMONIAL', 'ROA_ANTERIOR', 'GANANCIA_ANTERIOR'
    ]
    features_categoricas = ['MACROSECTOR', 'REGIÓN']
    
    X = df[features_numericas + features_categoricas]
    y = df['DELTA_GANANCIA']

    # 6. Pipeline
    # El SimpleImputer(strategy='median') se encargará de rellenar los NaNs que creamos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', StandardScaler())
            ]), features_numericas),
            ('cat', OneHotEncoder(handle_unknown='ignore'), features_categoricas)
        ])

    modelo_delta = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Entrenar
    print("🧠 Entrenando Random Forest...")
    modelo_delta.fit(X, y)
    
    # Guardar
    os.makedirs('models', exist_ok=True)
    ruta_guardado = 'models/modelo_delta_rf.pkl'
    joblib.dump(modelo_delta, ruta_guardado, compress=3)
    
    print(f"✅ ¡Éxito! Modelo guardado en: {ruta_guardado}")

if __name__ == "__main__":
    entrenar_modelo_delta()