import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

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

tab_macro, tab_micro = st.tabs(["📊 Pulso del Mercado (Macro)", "🚀 Simulador de Crecimiento (IA)"])

# ==============================================================================
# PESTAÑA 1: MACRO
# ==============================================================================
with tab_macro:
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