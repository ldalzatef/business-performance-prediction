import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import requests

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import procesar_datos_usuario, predecir_ganancia_futura

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Valora | Inteligencia Financiera", 
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: var(--secondary-background-color);
        border-left: 5px solid #636EFA;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNCIONES DE CARGA ---
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('models/modelo_delta_rf.pkl')
    except FileNotFoundError:
        return None

@st.cache_data
def cargar_datos_mercado():
    ruta_csv = 'data/greatest_colombian_business.csv'
    if not os.path.exists(ruta_csv):
        return None
    df = pd.read_csv(ruta_csv)
    
    # Limpieza Básica
    cols_num = ['INGRESOS OPERACIONALES', 'GANANCIA (PÉRDIDA)', 'TOTAL ACTIVOS', 'TOTAL PASIVOS', 'TOTAL PATRIMONIO']
    for col in cols_num:
        if df[col].dtype == 'O':
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Año de Corte' in df.columns:
        df['Año de Corte'] = pd.to_numeric(df['Año de Corte'].astype(str).str.replace(',', ''), errors='coerce')
    
    # --- CONVERSIÓN A MILLONES (VISUALIZACIÓN) ---
    # El dataset original está en "Miles de Millones". 
    # Multiplicamos por 1,000 para que todo el dashboard esté en "Millones".
    for col in cols_num:
        df[col] = df[col] * 1000

    # Ratios Globales (Estos no cambian con la escala)
    df['ROA'] = df['GANANCIA (PÉRDIDA)'] / df['TOTAL ACTIVOS'].replace(0, np.nan)
    df['ENDEUDAMIENTO'] = df['TOTAL PASIVOS'] / df['TOTAL ACTIVOS'].replace(0, np.nan)
    df['MARGEN_NETO'] = df['GANANCIA (PÉRDIDA)'] / df['INGRESOS OPERACIONALES'].replace(0, np.nan)
    
    return df

modelo_delta = cargar_modelo()
df_mercado = cargar_datos_mercado()

if not modelo_delta or df_mercado is None:
    st.error("⚠️ Error Crítico: No se encuentran los modelos o datos.")
    st.stop()

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.title("💎 Valora")
    st.caption("Predicción de Crecimiento Financiero")
    
    with st.expander("📖 Sobre la Herramienta"):
        st.write("""
        **Valora** utiliza Inteligencia Artificial para estimar la inercia de crecimiento.
        
        *Unidades:* Todas las cifras monetarias están expresadas en **Millones de Pesos (COP)**.
        """)
    
    st.divider()
    st.header("🌍 Configuración de Mercado")
    
    sectores = sorted(df_mercado['MACROSECTOR'].unique().astype(str))
    sector_sel = st.selectbox("Sector Económico", sectores, index=sectores.index('COMERCIO') if 'COMERCIO' in sectores else 0)
    
    regiones = sorted(df_mercado['REGIÓN'].unique().astype(str))
    region_sel = st.selectbox("Región Geográfica", regiones, index=regiones.index('Bogotá - Cundinamarca') if 'Bogotá - Cundinamarca' in regiones else 0)

    st.info("💡 Los gráficos se actualizarán según estos filtros.")

# --- LÓGICA DE FILTRADO ---
df_sector = df_mercado[df_mercado['MACROSECTOR'] == sector_sel]
df_filtrado = df_sector[df_sector['REGIÓN'] == region_sel]
anio_max = int(df_mercado['Año de Corte'].max())

# --- 4. PANEL PRINCIPAL ---
st.title(f"Panorama: {sector_sel}")
st.markdown(f"Análisis estratégico en **{region_sel}** (Cifras en Millones de COP).")

tab_global, tab_macro, tab_micro = st.tabs(["🌐 Análisis Global (Visión)", "📊 Pulso del Mercado (Macro)", "🚀 Simulador de Crecimiento (IA)"])

# ======================================================================
# PESTAÑA 1: GLOBAL (nuevo)
# ======================================================================
with tab_global:
    # Insertamos el expander global (copia del bloque original)
    # --- ANÁLISIS GLOBAL (Visión previa a filtros por sector/región) ---
    with st.expander("🌐 Análisis Global (Visión Agregada)", expanded=False):
        st.write("Resumen agregado de la muestra completa — por sector y por región. Puedes optar por usar las predicciones del modelo (más precisas, pero más lentas) o analizar cambios históricos.")

        use_model = st.checkbox("Usar predicciones del modelo para el análisis (más preciso, puede tardar)", value=False)

        geojson_path = os.path.join('data', 'colombia_departments.geojson')
        if not os.path.exists(geojson_path):
            if st.button("Descargar geojson de departamentos (opcional)"):
                # Lista de URLs alternativas (intentar en orden hasta encontrar una válida)
                candidatos = [
                    'https://raw.githubusercontent.com/marcovega/colombia-json/master/colombia_departments.geojson',
                    'https://raw.githubusercontent.com/kelvins/mas-utils/master/geojson/colombia_departamentos.geojson',
                    'https://raw.githubusercontent.com/martinzlopez/colombia-geojson/master/departamentos.geojson',
                    'https://raw.githubusercontent.com/juanelas/colombia-geojson/master/colombia_departments.geojson',
                ]
                ok = False
                for download_url in candidatos:
                    try:
                        r = requests.get(download_url, timeout=10)
                        r.raise_for_status()
                        with open(geojson_path, 'wb') as f:
                            f.write(r.content)
                        st.success(f"GeoJSON descargado correctamente desde: {download_url}")
                        ok = True
                        break
                    except Exception as ee:
                        # try next
                        last_exc = ee
                if not ok:
                    st.warning(f"No fue posible descargar el geojson: {last_exc}")

        @st.cache_data
        def preparar_analisis_global(df):
            dfg = df.copy()
            col_gan = 'GANANCIA (PÉRDIDA)'
            col_year = 'Año de Corte'

            dfg = dfg.sort_values(['NIT', col_year])
            dfg['GAN_ANT'] = dfg.groupby('NIT')[col_gan].shift(1)
            dfg['DELTA_ABS'] = dfg[col_gan] - dfg['GAN_ANT']
            dfg['DELTA_PCT'] = np.where(dfg['GAN_ANT'].abs() > 0, 100 * dfg['DELTA_ABS'] / dfg['GAN_ANT'].abs(), np.nan)

            agg_historico = (
                dfg.groupby('MACROSECTOR').agg(
                    patrimonio_med=('TOTAL PATRIMONIO', 'median'),
                    growth_med_pct=('DELTA_PCT', 'median'),
                    sum_delta_mill=('DELTA_ABS', 'sum'),
                    pct_negatives=('DELTA_ABS', lambda x: 100 * (x < 0).sum() / max(1, x.count())),
                    n_empresas=('NIT', 'nunique')
                )
                .reset_index()
                .dropna(subset=['patrimonio_med'])
            )

            latest_year = int(dfg[col_year].max())
            df_latest = dfg[dfg[col_year] == latest_year].copy()

            bins = [0, 50000, 200000, 1000000, np.inf]
            labels = ['Pequeña', 'Mediana', 'Grande', 'Mega-empresa']
            df_latest['SEGMENTO_ACTIVOS'] = pd.cut(df_latest['TOTAL ACTIVOS'], bins=bins, labels=labels)

            return agg_historico, df_latest, dfg

        @st.cache_data
        def calcular_predicciones_modelo(df_latest, _modelo):
            modelo_local = _modelo
            if modelo_local is None or df_latest.empty:
                return df_latest.assign(DELTA_PRED_MILL=np.nan, GAN_PRED_MILL=np.nan, GROWTH_PCT=np.nan)

            features_num = ['LOG_TOTAL ACTIVOS', 'LOG_INGRESOS OPERACIONALES', 'LOG_TOTAL PASIVOS',
                            'RATIO_ENDEUDAMIENTO', 'RATIO_PATRIMONIAL', 'ROA_ANTERIOR', 'GANANCIA_ANTERIOR']
            features_cat = ['MACROSECTOR', 'REGIÓN']

            dfm = df_latest.copy()
            factor = 1000.0
            dfm['LOG_TOTAL ACTIVOS'] = np.log1p(np.maximum(dfm['TOTAL ACTIVOS'] / factor, 0))
            dfm['LOG_INGRESOS OPERACIONALES'] = np.log1p(np.maximum(dfm['INGRESOS OPERACIONALES'] / factor, 0))
            dfm['LOG_TOTAL PASIVOS'] = np.log1p(np.maximum(dfm['TOTAL PASIVOS'] / factor, 0))

            # (El expander 'Análisis Global' fue movido a la pestaña principal.)

            # GANANCIA_ANTERIOR como la ganancia actual en unidades del modelo
            dfm['GANANCIA_ANTERIOR'] = (dfm['GANANCIA (PÉRDIDA)'] / factor)

            # Selección de columnas y limpieza
            cols_X = features_num + features_cat
            X = dfm.reindex(columns=cols_X)
            X = X.replace([np.inf, -np.inf], np.nan)

            try:
                delta_pred = modelo_local.predict(X)
            except Exception as e:
                # Si falla, devolvemos NaNs pero no rompemos la app
                st.warning(f"Error prediciendo con el modelo: {e}")
                return df_latest.assign(DELTA_PRED_MILL=np.nan, GAN_PRED_MILL=np.nan, GROWTH_PCT=np.nan)

            # delta_pred está en unidades del modelo (miles de millones) -> convertir a Millones
            delta_pred_mill = np.array(delta_pred) * factor
            gan_actual_mill = dfm['GANANCIA (PÉRDIDA)'].values
            gan_pred_mill = gan_actual_mill + delta_pred_mill
            growth_pct = np.where(np.abs(gan_actual_mill) > 0, 100 * delta_pred_mill / np.abs(gan_actual_mill), np.nan)

            df_out = df_latest.copy()
            df_out['DELTA_PRED_MILL'] = delta_pred_mill
            df_out['GAN_PRED_MILL'] = gan_pred_mill
            df_out['GROWTH_PCT'] = growth_pct
            return df_out

        try:
            agg_sector_hist, df_latest_global, df_hist = preparar_analisis_global(df_mercado)
        except Exception as e:
            st.error(f"Error preparando análisis global: {e}")
            agg_sector_hist = pd.DataFrame()
            df_latest_global = pd.DataFrame()
            df_hist = pd.DataFrame()

        # Si el usuario quiere usar el modelo, calcular predicciones cacheadas
        if use_model and not df_latest_global.empty:
            with st.spinner('Calculando predicciones del modelo (cacheadas)...'):
                df_preds = calcular_predicciones_modelo(df_latest_global, modelo_delta)
                # Agregados por sector usando predicción
                agg_sector = (
                    df_preds.groupby('MACROSECTOR').agg(
                        patrimonio_med=('TOTAL PATRIMONIO', 'median'),
                        growth_med_pct=('GROWTH_PCT', 'median'),
                        sum_delta_mill=('DELTA_PRED_MILL', 'sum'),
                        pct_negatives=('DELTA_PRED_MILL', lambda x: 100 * (x < 0).sum() / max(1, x.count())),
                        n_empresas=('NIT', 'nunique')
                    ).reset_index().dropna(subset=['patrimonio_med'])
                )
        else:
            agg_sector = agg_sector_hist

        # 1) Matriz Sectorial (burbuja)
        if not agg_sector.empty:
            agg_sector['color'] = np.where(agg_sector['growth_med_pct'] > 5, 'green', np.where(agg_sector['growth_med_pct'] > 0, 'gold', 'red'))
            # Asegurar que el tamaño de la burbuja sea no-negativo
            agg_sector['size_val'] = agg_sector['sum_delta_mill'].abs()
            # Evitar ceros absolutos para que Plotly no reciba tamaños negativos o 0
            min_size = agg_sector['size_val'].replace(0, np.nan).min()
            if pd.isna(min_size):
                # Si todo es 0 o NaN, dar un tamaño constante
                agg_sector['size_val'] = 1.0
            else:
                # Remplazar 0s por una fracción pequeña del mínimo positivo
                agg_sector['size_val'] = agg_sector['size_val'].replace(0, min_size * 0.1)

            fig_mat = px.scatter(
                agg_sector.sort_values('size_val', ascending=False),
                x='patrimonio_med', y='growth_med_pct', size='size_val', color='color',
                hover_name='MACROSECTOR', hover_data={'patrimonio_med':':,.0f','growth_med_pct':':.2f','sum_delta_mill':':,.0f','n_empresas':True},
                labels={'patrimonio_med': 'Patrimonio Mediano (Millones)', 'growth_med_pct': 'Δ Ganancia Mediana (%)'},
                title='Matriz de Potencial y Rentabilidad Sectorial', size_max=80
            )
            fig_mat.update_xaxes(type='log')
            st.plotly_chart(fig_mat, use_container_width=True)
        else:
            st.info('No hay datos suficientes para la matriz sectorial.')

        # 2) Dispersión Patrimonio vs Ganancia (log X)
        if not df_latest_global.empty:
            df_sc = df_latest_global.dropna(subset=['TOTAL PATRIMONIO', 'GANANCIA (PÉRDIDA)'])
            if not df_sc.empty:
                # si usamos predicciones, pinta esas
                if use_model and 'GAN_PRED_MILL' in locals():
                    df_sc_plot = df_preds.copy()
                    y_col = 'GAN_PRED_MILL'
                    y_label = 'Ganancia Predicha (Millones)'
                else:
                    df_sc_plot = df_sc
                    y_col = 'GANANCIA (PÉRDIDA)'
                    y_label = 'Ganancia (Millones)'

                # Muestreo para mantener responsividad
                if len(df_sc_plot) > 5000:
                    df_sc_plot = df_sc_plot.sample(5000, random_state=42)
                    st.caption('Muestra aleatoria mostrada (5,000 observaciones) para rapidez.')

                x = df_sc_plot['TOTAL PATRIMONIO'].replace(0, np.nan).dropna()
                try:
                    coef = np.polyfit(np.log10(x), df_sc_plot.loc[x.index, y_col], 2)
                    fit_x = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
                    fit_y = np.polyval(coef, np.log10(fit_x))
                except Exception:
                    fit_x = []
                    fit_y = []

                fig_disp = px.scatter(df_sc_plot, x='TOTAL PATRIMONIO', y=y_col, opacity=0.5,
                                      labels={'TOTAL PATRIMONIO':'Patrimonio (Millones)', y_col:y_label},
                                      hover_data=['RAZÓN SOCIAL','MACROSECTOR','REGIÓN'],
                                      title='Proyección Ganancias vs. Solidez Patrimonial (Último Año, escala log)')
                fig_disp.update_xaxes(type='log')
                if len(fit_x):
                    fig_disp.add_traces(px.line(x=fit_x, y=fit_y).data)
                st.plotly_chart(fig_disp, use_container_width=True)
            else:
                st.info('No hay datos para la dispersión global.')

        # 3) Boxplot por tamaño
        if not df_latest_global.empty:
            df_box = df_latest_global.dropna(subset=['SEGMENTO_ACTIVOS', 'DELTA_PCT'])
            if use_model and not df_preds.empty:
                df_box = df_preds.dropna(subset=['SEGMENTO_ACTIVOS','GROWTH_PCT']).copy()
                # Evitar duplicar la columna 'DELTA_PCT' (ya existe en df_latest). Reemplazamos/creamos columna única.
                if 'DELTA_PCT' in df_box.columns:
                    df_box.drop(columns=['DELTA_PCT'], inplace=True)
                df_box = df_box.rename(columns={'GROWTH_PCT': 'DELTA_PCT'})

            if not df_box.empty:
                fig_box = px.box(df_box, x='SEGMENTO_ACTIVOS', y='DELTA_PCT', points='outliers',
                                 labels={'SEGMENTO_ACTIVOS':'Segmento','DELTA_PCT':'Δ Ganancia (%)'},
                                 title='Distribución de Crecimiento (Análisis de Colas) por Tamaño')
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info('No hay datos suficientes para boxplots por tamaño.')

        # 4) Mapa Choropleth por Región (si existe geojson)
        if os.path.exists(geojson_path) and not df_hist.empty:
            import json
            try:
                with open(geojson_path, 'r', encoding='utf-8') as f:
                    geo = json.load(f)
                if use_model and not df_preds.empty:
                    reg_agg = df_preds.groupby('REGIÓN').apply(lambda d: 100 * (d['DELTA_PRED_MILL'] < 0).sum() / max(1, d['DELTA_PRED_MILL'].count())).reset_index(name='pct_neg')
                else:
                    reg_agg = df_hist.groupby('REGIÓN').apply(lambda d: 100 * (d['DELTA_ABS'] < 0).sum() / max(1, d['DELTA_ABS'].count())).reset_index(name='pct_neg')

                fig_map = px.choropleth(reg_agg, geojson=geo, locations='REGIÓN', color='pct_neg', featureidkey='properties.name',
                                        color_continuous_scale='YlOrRd', title='Porcentaje de Empresas con Δ Ganancia Negativa (%) por Región')
                fig_map.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.warning(f"Error al renderizar mapa: {e}")
        else:
            st.info("Para mostrar el mapa, añade 'data/colombia_departments.geojson' con geometrías de departamentos o utiliza el botón de descarga.")

        st.markdown('---')

    st.markdown("### 1. Indicadores Clave (KPIs)")
    
    df_actual = df_filtrado[df_filtrado['Año de Corte'] == anio_max]
    
    if df_actual.empty:
        st.warning("No hay datos suficientes.")
    else:
        # Cálculos (Ya están en Millones gracias a la carga de datos)
        roa_prom = df_actual['ROA'].median() * 100
        deuda_prom = df_actual['ENDEUDAMIENTO'].median() * 100
        margen_prom = df_actual['MARGEN_NETO'].median() * 100
        ingresos_prom = df_actual['INGRESOS OPERACIONALES'].median() 
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROA Típico", f"{roa_prom:.1f}%", help="Eficiencia (Mediana)")
        col2.metric("Nivel de Deuda", f"{deuda_prom:.1f}%", delta_color="inverse")
        col3.metric("Margen Neto", f"{margen_prom:.1f}%")
        col4.metric("Ingresos Típicos", f"${ingresos_prom:,.0f} M", help="Millones de COP")
        
        st.markdown("---")

        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            st.subheader("Mapa de Riesgo vs. Retorno")
            # Scatter: Eje X=Endeudamiento, Eje Y=ROA
            fig_risk = px.scatter(
                df_actual, 
                x='ENDEUDAMIENTO', 
                y='ROA', 
                size='TOTAL ACTIVOS', # El tamaño de burbuja usa Activos en Millones
                color='GANANCIA (PÉRDIDA)',
                hover_name='RAZÓN SOCIAL',
                range_x=[0, 1.2], 
                range_y=[-0.5, 0.5],
                color_continuous_scale='Viridis',
                title=f"Eficiencia vs. Endeudamiento ({anio_max})"
            )
            fig_risk.update_layout(xaxis_title="Endeudamiento", yaxis_title="ROA")
            st.plotly_chart(fig_risk, use_container_width=True)
            
        with c2:
            st.subheader("Distribución de Ganancias")
            fig_hist = px.box(
                df_actual, 
                y='GANANCIA (PÉRDIDA)', 
                points="outliers",
                title=f"Rango de Utilidades (Millones)",
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.subheader(f"🏆 Top 10 Líderes (Ingresos en Millones)")
        top_10 = df_actual.nlargest(10, 'INGRESOS OPERACIONALES')[['RAZÓN SOCIAL', 'INGRESOS OPERACIONALES', 'ROA', 'MARGEN_NETO']]
        
        st.dataframe(
            top_10.style.format({
                'INGRESOS OPERACIONALES': '${:,.0f}', 
                'ROA': '{:.1%}', 
                'MARGEN_NETO': '{:.1%}'
            }), 
            use_container_width=True
        )

# ==============================================================================
# PESTAÑA 2: MICRO (SIMULADOR)
# ==============================================================================
with tab_micro:
    st.markdown("### 🔮 Proyección Financiera")
    st.caption("Ingresa los datos de tu empresa en **Millones de Pesos**.")
    
    with st.container(border=True):
        col_in1, col_in2, col_in3 = st.columns(3)
        
        # INPUTS EN MILLONES (Valores por defecto ajustados: 50,000 Millones = 50 Miles de Millones)
        with col_in1:
            st.markdown("#### 1. Tamaño")
            activos_mill = st.number_input("Total Activos ($M)", min_value=1.0, value=50000.0, step=1000.0)
            patrimonio_mill = st.number_input("Total Patrimonio ($M)", value=20000.0, step=1000.0)
            
        with col_in2:
            st.markdown("#### 2. Operación")
            ingresos_mill = st.number_input("Ingresos Operacionales ($M)", min_value=0.0, value=60000.0, step=1000.0)
            pasivos_mill = st.number_input("Total Pasivos ($M)", min_value=0.0, value=30000.0, step=1000.0)
            
        with col_in3:
            st.markdown("#### 3. Historia")
            ganancia_actual_mill = st.number_input("Ganancia Año Actual ($M)", value=5000.0, step=500.0)
            
        btn_predecir = st.button("Simular Crecimiento ⚡", type="primary", use_container_width=True)

    if btn_predecir:
        # --- TRADUCCIÓN PARA LA IA ---
        # El modelo fue entrenado en "Miles de Millones".
        # Debemos dividir los inputs del usuario por 1,000 antes de enviarlos al modelo.
        factor_conv = 1000.0
        
        inputs_modelo = {
            'TOTAL ACTIVOS': activos_mill / factor_conv,
            'TOTAL PASIVOS': pasivos_mill / factor_conv,
            'TOTAL PATRIMONIO': patrimonio_mill / factor_conv,
            'INGRESOS OPERACIONALES': ingresos_mill / factor_conv,
            'GANANCIA_ANTERIOR': ganancia_actual_mill / factor_conv,
            'MACROSECTOR': sector_sel,
            'REGIÓN': region_sel
        }
        
        try:
            # 1. Procesar y Predecir (El modelo devuelve Miles de Millones)
            df_procesado = procesar_datos_usuario(inputs_modelo)
            ganancia_futura_mm, delta_predicho_mm = predecir_ganancia_futura(modelo_delta, df_procesado, inputs_modelo['GANANCIA_ANTERIOR'])
            
            # 2. Convertir Resultado de vuelta a Millones para mostrar al usuario
            delta_predicho_mill = delta_predicho_mm * factor_conv
            ganancia_futura_mill = ganancia_futura_mm * factor_conv
            
            # Crecimiento Porcentual (No necesita conversión)
            crecimiento_pct = (delta_predicho_mm / abs(inputs_modelo['GANANCIA_ANTERIOR'])) * 100 if inputs_modelo['GANANCIA_ANTERIOR'] != 0 else 0
            
            # --- RESULTADOS ---
            st.divider()
            
            # Diagnóstico
            roa_usuario = (ganancia_actual_mill / activos_mill) * 100
            st.caption(f"Diagnóstico: Tu ROA actual es del **{roa_usuario:.1f}%**.")
            
            c_res1, c_res2, c_res3 = st.columns(3)
            
            # Mostramos todo en Millones
            c_res1.metric("Proyección (Año Siguiente)", f"${ganancia_futura_mill:,.0f} M", 
                          delta=f"{delta_predicho_mill:,.0f} M (Cambio)")
            
            c_res2.metric("Crecimiento Esperado", f"{crecimiento_pct:+.1f}%")
            
            estatus = "Expansión" if delta_predicho_mill > 0 else "Contracción"
            c_res3.metric("Tendencia", estatus, delta="Positiva" if delta_predicho_mill > 0 else "Negativa")
            
            # Gráfico Cascada
            st.subheader("Puente de Valor (Millones COP)")
            
            fig_waterfall = go.Figure(go.Waterfall(
                name = "Flujo", orientation = "v",
                measure = ["absolute", "relative", "total"],
                x = ["Actual", "Variación", "Proyectado"],
                textposition = "outside",
                # Formateamos texto en el gráfico para que se vea limpio
                text = [f"{ganancia_actual_mill:,.0f}", f"{delta_predicho_mill:+,.0f}", f"{ganancia_futura_mill:,.0f}"],
                y = [ganancia_actual_mill, delta_predicho_mill, ganancia_futura_mill],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            fig_waterfall.update_layout(title = "Evolución de Ganancias", showlegend = False)
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error en el cálculo: {e}")
        st.markdown("---")
        st.caption("Fuente de Datos:")
        st.link_button("🔗 Ver Datos Abiertos (Supersociedades)", "https://www.datos.gov.co/Comercio-Industria-y-Turismo/10-000-Empresas-mas-Grandes-del-Pa-s/6cat-2gcs/about_data")