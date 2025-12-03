import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Agregar path para importar utils
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
    /* Tarjetas de métricas */
    .metric-card {
        background-color: var(--secondary-background-color);
        border-left: 5px solid #636EFA;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    /* FOOTER ESTÁTICO */
    .footer {
        width: 100%;
        margin-top: 50px;
        padding: 30px 0px;
        border-top: 1px solid var(--secondary-background-color);
        text-align: center;
        color: var(--text-color);
        background-color: var(--background-color);
    }
    
    .footer-content {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        flex-wrap: wrap;
    }
    
    .footer-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
    }
    
    .footer a {
        color: #636EFA;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.3s;
    }
    
    .footer a:hover {
        color: #00CC96;
        text-decoration: underline;
    }
    
    .team-title {
        font-size: 12px;
        color: gray;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Ajuste para que el footer no tape el contenido final */
    .block-container { padding-bottom: 50px; }
</style>
""", unsafe_allow_html=True)

# --- 2. FUNCIONES DE CARGA Y CÁLCULO ---

def calcular_ratios(df):
    """Calcula ratios financieros básicos sobre un DataFrame dado"""
    df['ROA'] = df['GANANCIA (PÉRDIDA)'] / df['TOTAL ACTIVOS'].replace(0, np.nan)
    df['ENDEUDAMIENTO'] = df['TOTAL PASIVOS'] / df['TOTAL ACTIVOS'].replace(0, np.nan)
    df['MARGEN_NETO'] = df['GANANCIA (PÉRDIDA)'] / df['INGRESOS OPERACIONALES'].replace(0, np.nan)
    df['ROTACION_ACTIVOS'] = df['INGRESOS OPERACIONALES'] / df['TOTAL ACTIVOS'].replace(0, np.nan)
    return df

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
    
    cols_num = ['INGRESOS OPERACIONALES', 'GANANCIA (PÉRDIDA)', 'TOTAL ACTIVOS', 'TOTAL PASIVOS', 'TOTAL PATRIMONIO']
    for col in cols_num:
        if df[col].dtype == 'O':
            df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Año de Corte' in df.columns:
        df['Año de Corte'] = pd.to_numeric(df['Año de Corte'].astype(str).str.replace(',', ''), errors='coerce')
    
    # CONVERSIÓN A MILLONES
    for col in cols_num:
        df[col] = df[col] * 1000

    df = calcular_ratios(df)
    return df

def simular_futuro_sectores(df_actual, modelo):
    df_sectores = df_actual.groupby('MACROSECTOR').agg({
        'TOTAL ACTIVOS': 'median', 'TOTAL PASIVOS': 'median',
        'TOTAL PATRIMONIO': 'median', 'INGRESOS OPERACIONALES': 'median',
        'GANANCIA (PÉRDIDA)': 'median',
    }).reset_index()
    
    nuevas_ganancias = []
    for _, row in df_sectores.iterrows():
        inputs = {
            'TOTAL ACTIVOS': row['TOTAL ACTIVOS'] / 1000,
            'TOTAL PASIVOS': row['TOTAL PASIVOS'] / 1000,
            'TOTAL PATRIMONIO': row['TOTAL PATRIMONIO'] / 1000,
            'INGRESOS OPERACIONALES': row['INGRESOS OPERACIONALES'] / 1000,
            'GANANCIA_ANTERIOR': row['GANANCIA (PÉRDIDA)'] / 1000,
            'MACROSECTOR': row['MACROSECTOR'], 'REGIÓN': 'Bogotá - Cundinamarca'
        }
        try:
            df_proc = procesar_datos_usuario(inputs)
            delta_pred = modelo.predict(df_proc)[0]
            ganancia_futura = (inputs['GANANCIA_ANTERIOR'] + delta_pred) * 1000 
            nuevas_ganancias.append(ganancia_futura)
        except: nuevas_ganancias.append(row['GANANCIA (PÉRDIDA)'])
            
    df_sectores['GANANCIA (PÉRDIDA)'] = nuevas_ganancias
    df_sectores = calcular_ratios(df_sectores)
    return df_sectores

def simular_futuro_empresas(df_empresas, modelo):
    nuevas_ganancias = []
    for _, row in df_empresas.iterrows():
        inputs = {
            'TOTAL ACTIVOS': row['TOTAL ACTIVOS'] / 1000,
            'TOTAL PASIVOS': row['TOTAL PASIVOS'] / 1000,
            'TOTAL PATRIMONIO': row['TOTAL PATRIMONIO'] / 1000,
            'INGRESOS OPERACIONALES': row['INGRESOS OPERACIONALES'] / 1000,
            'GANANCIA_ANTERIOR': row['GANANCIA (PÉRDIDA)'] / 1000,
            'MACROSECTOR': row['MACROSECTOR'], 'REGIÓN': row['REGIÓN']
        }
        try:
            df_proc = procesar_datos_usuario(inputs)
            delta_pred = modelo.predict(df_proc)[0]
            ganancia_futura = (inputs['GANANCIA_ANTERIOR'] + delta_pred) * 1000
            nuevas_ganancias.append(ganancia_futura)
        except:
            nuevas_ganancias.append(row['GANANCIA (PÉRDIDA)'])
    
    df_simulado = df_empresas.copy()
    df_simulado['GANANCIA (PÉRDIDA)'] = nuevas_ganancias
    df_simulado = calcular_ratios(df_simulado)
    return df_simulado

modelo_delta = cargar_modelo()
df_mercado = cargar_datos_mercado()

if not modelo_delta or df_mercado is None:
    st.error("⚠️ Error Crítico: No se encuentran los modelos o datos.")
    st.stop()

# --- 3. SIDEBAR ---
with st.sidebar:
    # 1. LOGO DE LA APP
    # Asegúrate de que el archivo esté en la carpeta app/
    logo_path = "app/LOGO VALORA.png" 
    
    if os.path.exists(logo_path):
        st.image(logo_path, width=180) 
    else:
        st.markdown("<h1 style='text-align: center;'>💎</h1>", unsafe_allow_html=True)

    st.title("Valora")
    st.caption("Predicción de Crecimiento Financiero")
    
    with st.expander("📖 Fuente de Datos"):
        st.write("Análisis basado en las 10,000 empresas más grandes de Colombia.")
        st.markdown("[🔗 Ver Datos Abiertos (Gov.co)](https://www.datos.gov.co/Comercio-Industria-y-Turismo/10-000-Empresas-mas-Grandes-del-Pa-s/6cat-2gcs/about_data)")
    
    st.divider()
    st.header("🌍 Filtros de Detalle")
    
    sectores = sorted(df_mercado['MACROSECTOR'].unique().astype(str))
    sector_sel = st.selectbox("Sector Económico", sectores, index=sectores.index('COMERCIO') if 'COMERCIO' in sectores else 0)
    
    regiones = sorted(df_mercado['REGIÓN'].unique().astype(str))
    region_sel = st.selectbox("Región / Dpto.", regiones, index=regiones.index('Bogotá - Cundinamarca') if 'Bogotá - Cundinamarca' in regiones else 0)

# --- 4. PANEL PRINCIPAL ---
st.title("Panorama Empresarial Colombiano")

tab_global, tab_sector, tab_simulador = st.tabs([
    "🌍 Visión Global IA", 
    "📊 Contexto Sectorial", 
    "🔮 Simulador IA"
])

anio_max = int(df_mercado['Año de Corte'].max())
df_historico_anio = df_mercado[df_mercado['Año de Corte'] == anio_max].copy()
df_historico_anio = calcular_ratios(df_historico_anio)

# ==============================================================================
# PESTAÑA 1: VISIÓN GLOBAL
# ==============================================================================
with tab_global:
    col_header, col_toggle = st.columns([3, 1])
    with col_header:
        st.markdown(f"### 🇨🇴 Competitividad Nacional")
        st.caption("Análisis comparativo de todos los sectores.")
    with col_toggle:
        ver_futuro_global = st.toggle("🔮 Proyección IA (2025)", value=False, key="toggle_global")
    
    if ver_futuro_global:
        df_visual = simular_futuro_sectores(df_historico_anio, modelo_delta)
        estado_txt = "PROYECTADO (IA)"
    else:
        df_visual = df_historico_anio.groupby('MACROSECTOR').agg({
            'TOTAL ACTIVOS': 'median', 'TOTAL PASIVOS': 'median',
            'INGRESOS OPERACIONALES': 'median', 'GANANCIA (PÉRDIDA)': 'median',
            'ROA': 'median', 'ENDEUDAMIENTO': 'median',
            'ROTACION_ACTIVOS': 'median', 'MARGEN_NETO': 'median'
        }).reset_index()
        estado_txt = "HISTÓRICO (REAL)"

    volumenes = df_historico_anio.groupby('MACROSECTOR')['INGRESOS OPERACIONALES'].sum().reset_index()
    df_visual = df_visual.merge(volumenes, on='MACROSECTOR', suffixes=('', '_TOTAL'))
    df_visual['VOLUMEN_TOTAL'] = df_visual['INGRESOS OPERACIONALES_TOTAL']

    # KPIs GLOBALES
    roa_pais = df_visual['ROA'].median() * 100
    margen_pais = df_visual['MARGEN_NETO'].median() * 100
    deuda_pais = df_visual['ENDEUDAMIENTO'].median() * 100
    
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Eficiencia País ({estado_txt})", f"{roa_pais:.2f}%")
    k2.metric("Riesgo País (Deuda)", f"{deuda_pais:.2f}%", delta_color="inverse")
    k3.metric("Margen Neto Promedio", f"{margen_pais:.2f}%")
    st.divider()

    st.subheader("1. Mapa de Riesgo vs. Retorno")
    df_visual_bubbles = df_visual.sort_values(by='VOLUMEN_TOTAL', ascending=False)
    fig_bubbles = px.scatter(
        df_visual_bubbles, x="ENDEUDAMIENTO", y="ROA", size="VOLUMEN_TOTAL", color="MACROSECTOR",
        hover_name="MACROSECTOR", text="MACROSECTOR", size_max=70,
        title=f"Posición Estratégica ({estado_txt})",
        labels={"ENDEUDAMIENTO": "Deuda", "ROA": "Rentabilidad"}
    )
    fig_bubbles.update_traces(textposition='middle center', textfont=dict(color='white', size=10))
    fig_bubbles.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_bubbles.update_layout(showlegend=False, height=550)
    st.plotly_chart(fig_bubbles, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("2. Mapa de Calor del Mercado")
    df_tree = df_visual[df_visual['GANANCIA (PÉRDIDA)'] > 0].copy()
    if not df_tree.empty:
        fig_tree = px.treemap(
            df_tree, path=['MACROSECTOR'], values='VOLUMEN_TOTAL', color='GANANCIA (PÉRDIDA)',
            color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
            title=f"Dominio de Mercado y Rentabilidad ({estado_txt})"
        )
        fig_tree.update_layout(height=500)
        st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("3. Velocidad del Negocio")
    df_visual_sorted = df_visual.sort_values(by='ROTACION_ACTIVOS', ascending=False)
    fig_bar = px.bar(
        df_visual_sorted, x="ROTACION_ACTIVOS", y="MACROSECTOR", orientation='h',
        text_auto='.2f', color="MACROSECTOR", title=f"Eficiencia Operativa ({estado_txt})"
    )
    fig_bar.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig_bar, use_container_width=True)


# ==============================================================================
# PESTAÑA 2: CONTEXTO SECTORIAL
# ==============================================================================
with tab_sector:
    col_head_sec, col_toggle_sec = st.columns([3, 1])
    with col_head_sec:
        st.markdown(f"### 📊 Análisis del Sector: **{sector_sel}**")
        st.caption(f"Comparativa Nacional vs. Regional ({anio_max})")
    with col_toggle_sec:
        ver_futuro_sector = st.toggle("🔮 Simular Empresas (IA)", value=False, key="toggle_sector")

    df_sector_nacional_base = df_historico_anio[df_historico_anio['MACROSECTOR'] == sector_sel].copy()
    
    if ver_futuro_sector:
        st.toast(f"Proyectando el futuro de {len(df_sector_nacional_base)} empresas...", icon="🚀")
        with st.spinner('La IA está analizando empresa por empresa...'):
            df_sector_nacional = simular_futuro_empresas(df_sector_nacional_base, modelo_delta)
        estado_sec_txt = "PROYECTADO 2025"
    else:
        df_sector_nacional = df_sector_nacional_base
        estado_sec_txt = "HISTÓRICO REAL"
        
    df_sector_regional = df_sector_nacional[df_sector_nacional['REGIÓN'] == region_sel].copy()

    st.markdown(f"#### 1. Radiografía Nacional ({estado_sec_txt})")
    st.caption("Comportamiento de todas las empresas del sector en el país.")
    
    df_scatter = df_sector_nacional[(df_sector_nacional['ROA'] > -0.8) & (df_sector_nacional['ROA'] < 0.8)]
    fig_companies = px.scatter(
        df_scatter, x="INGRESOS OPERACIONALES", y="ROA", color="REGIÓN",
        hover_name="RAZÓN SOCIAL", log_x=True, 
        title=f"Dispersión: Tamaño vs. Eficiencia ({estado_sec_txt})",
        labels={"INGRESOS OPERACIONALES": "Ingresos (Log)", "ROA": "Rentabilidad (ROA)"}
    )
    fig_companies.update_layout(height=500)
    st.plotly_chart(fig_companies, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    metric_dona = 'GANANCIA (PÉRDIDA)' if ver_futuro_sector else 'INGRESOS OPERACIONALES'
    label_dona = 'Ganancias Proyectadas' if ver_futuro_sector else 'Ingresos Reales'
    df_share = df_sector_nacional[df_sector_nacional[metric_dona] > 0].groupby('REGIÓN')[metric_dona].sum().reset_index()
    fig_donut = px.pie(
        df_share, values=metric_dona, names='REGIÓN', hole=0.4,
        title=f"Participación Regional por {label_dona}"
    )
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig_donut, use_container_width=True)

    st.divider()

    st.markdown(f"#### 2. Benchmarking: {region_sel} vs. País")
    if df_sector_regional.empty:
        st.warning(f"No hay empresas en {region_sel} para este sector.")
    else:
        roa_nac = df_sector_nacional['ROA'].median() * 100
        roa_reg = df_sector_regional['ROA'].median() * 100
        margen_nac = df_sector_nacional['MARGEN_NETO'].median() * 100
        margen_reg = df_sector_regional['MARGEN_NETO'].median() * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"ROA Regional ({estado_sec_txt})", f"{roa_reg:.2f}%", delta=f"{roa_reg - roa_nac:.2f}% vs País")
        c2.metric("Margen Neto", f"{margen_reg:.2f}%", delta=f"{margen_reg - margen_nac:.2f}% vs País")
        c3.metric("Empresas Analizadas", f"{len(df_sector_regional)}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"🏆 Top 10 Líderes en {region_sel}")
        top_10 = df_sector_regional.nlargest(10, 'INGRESOS OPERACIONALES')[['RAZÓN SOCIAL', 'INGRESOS OPERACIONALES', 'GANANCIA (PÉRDIDA)', 'ROA']]
        st.dataframe(
            top_10,
            column_config={
                "INGRESOS OPERACIONALES": st.column_config.ProgressColumn("Ingresos ($M)", format="$%d M", min_value=0, max_value=top_10['INGRESOS OPERACIONALES'].max()),
                "GANANCIA (PÉRDIDA)": st.column_config.NumberColumn("Ganancia ($M)", format="$%d M"),
                "ROA": st.column_config.NumberColumn("ROA", format="%.2f%%")
            },
            hide_index=True, use_container_width=True
        )


# ==============================================================================
# PESTAÑA 3: SIMULADOR IA
# ==============================================================================
with tab_simulador:
    st.markdown("### 🔮 Valora Predictor")
    st.markdown("#### 🛡️ Transparencia del Modelo IA")
    k1, k2, k3 = st.columns(3)
    k1.metric("Certeza del Modelo (R²)", "77.4%", help="El modelo explica el 77.4% de los patrones de crecimiento.")
    k2.metric("Base de Conocimiento", "26k+ Empresas", help="Entrenado con datos reales de Supersociedades.")
    k3.markdown("**Interpretación:**<br>Estimación de inercia financiera basada en patrones históricos.", unsafe_allow_html=True)
    st.divider()

    with st.container(border=True):
        st.markdown("**Ingresa los datos de tu empresa:**")
        c1, c2, c3 = st.columns(3)
        with c1: 
            activos_mill = st.number_input("Total Activos ($M)", min_value=1.0, value=50000.0, step=1000.0)
            patrimonio_mill = st.number_input("Total Patrimonio ($M)", value=20000.0, step=1000.0)
        with c2:
            ingresos_mill = st.number_input("Ingresos ($M)", min_value=0.0, value=60000.0, step=1000.0)
            pasivos_mill = st.number_input("Pasivos ($M)", min_value=0.0, value=30000.0, step=1000.0)
        with c3:
            ganancia_actual_mill = st.number_input("Ganancia Actual ($M)", value=5000.0, step=500.0)
        btn_predecir = st.button("Simular Escenario ⚡", type="primary", use_container_width=True)

    if btn_predecir:
        factor = 1000.0
        inputs = {'TOTAL ACTIVOS': activos_mill/factor, 'TOTAL PASIVOS': pasivos_mill/factor, 
                  'TOTAL PATRIMONIO': patrimonio_mill/factor, 'INGRESOS OPERACIONALES': ingresos_mill/factor, 
                  'GANANCIA_ANTERIOR': ganancia_actual_mill/factor, 'MACROSECTOR': sector_sel, 'REGIÓN': region_sel}
        try:
            df_proc = procesar_datos_usuario(inputs)
            fut, delta = predecir_ganancia_futura(modelo_delta, df_proc, inputs['GANANCIA_ANTERIOR'])
            
            st.markdown("#### 🎯 Resultados de la Proyección")
            c1, c2, c3 = st.columns(3)
            c1.metric("Proyección", f"${fut*factor:,.0f} M", delta=f"{delta*factor:,.0f} M")
            c2.metric("Crecimiento", f"{(delta/abs(inputs['GANANCIA_ANTERIOR']))*100:+.2f}%")
            c3.metric("Tendencia", "Expansión" if delta>0 else "Contracción", delta=delta*factor, delta_color="normal")
            
            fig_w = go.Figure(go.Waterfall(name="20", orientation="v", measure=["absolute", "relative", "total"],
                x=["Actual", "Impacto IA", "Proyección"], y=[ganancia_actual_mill, delta*factor, fut*factor],
                text=[f"{ganancia_actual_mill:,.0f}", f"{delta*factor:+,.0f}", f"{fut*factor:,.0f}"], connector={"line":{"color":"#636EFA"}}))
            st.plotly_chart(fig_w, use_container_width=True)
        except Exception as e: st.error(f"Error: {e}")

# ==============================================================================
# FOOTER ESTÁTICO (EQUIPO)
# ==============================================================================
st.markdown("""
<div class="footer">
    <div class="team-title">Equipo de Desarrollo</div>
    <div class="footer-content">
        <div class="footer-item">
            🚀 <b>Valora App</b>
        </div>
        <div class="footer-item">|</div>
        <div class="footer-item">
            <a href="https://www.linkedin.com/in/ldalzatef/" target="_blank">👩‍💻 Leidy Alzate</a>
        </div>
        <div class="footer-item">
            <a href="https://www.linkedin.com/in/josuehernandezbautista/" target="_blank">🧑‍💻 Josué Hernández</a>
        </div>
        <div class="footer-item">
            <a href="https://www.linkedin.com/in/juanjosecifuentesrodriguez/" target="_blank">🧑‍💻 Juan Cifuentes</a>
        </div>
    </div>
    <div style="margin-top: 10px; font-size: 12px; color: gray;">
        Proyecto desarrollado para fines educativos y de investigación en el marco del concurso Datos al Ecosistema 2025.
    </div>
    <div style="margin-top: 10px; font-size: 12px; color: gray;">
        © 2025 Valora App. Todos los derechos reservados.
    </div>
        <div style="margin-top: 10px; font-size: 12px; color: gray;">
        versión 1.0.0
    </div>
</div>
""", unsafe_allow_html=True)