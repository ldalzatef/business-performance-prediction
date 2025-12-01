import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# 1. Configuración de la página
st.set_page_config(
    page_title="Predicción Financiera Colombia", 
    page_icon="💰",
    layout="wide"
)

# --- FUNCIONES DE CARGA (CACHÉ) ---
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('models/modelo_ganancias_rf.pkl')
    except FileNotFoundError:
        return None

@st.cache_data
def cargar_datos_historicos():
    ruta_csv = 'data/greatest_colombian_business.csv'
    if not os.path.exists(ruta_csv):
        return None

    df = pd.read_csv(ruta_csv)
    
    # Limpieza básica
    cols_numericas = ['INGRESOS OPERACIONALES', 'GANANCIA (PÉRDIDA)', 'TOTAL ACTIVOS', 'TOTAL PASIVOS', 'TOTAL PATRIMONIO']
    for col in cols_numericas:
        if df[col].dtype == 'O':
            df[col] = df[col].astype(str).str.replace('$', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    if 'Año de Corte' in df.columns:
        df['Año de Corte'] = pd.to_numeric(df['Año de Corte'].astype(str).str.replace(',', ''), errors='coerce')
        
    return df

modelo = cargar_modelo()
df_historico = cargar_datos_historicos()

if modelo is None or df_historico is None:
    st.error("⚠️ Error: Faltan archivos críticos (modelo .pkl o datos .csv).")
    st.stop()

# --- SIDEBAR (Solo Inputs) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3310/3310638.png", width=50)
    st.title("Parámetros")
    
    mostrar_ecopetrol = st.toggle("Incluir Casos Atípicos (Ecopetrol)", value=False)
    
    st.divider()

    with st.form("formulario_prediccion"):
        st.subheader("Datos Financieros")
        st.caption("Cifras en Miles de Millones de COP")
        
        ingresos = st.number_input("Ingresos Operacionales", min_value=0.0, value=15.0, step=1.0)
        activos = st.number_input("Total Activos", min_value=0.0, value=20.0, step=1.0)
        pasivos = st.number_input("Total Pasivos", min_value=0.0, value=10.0, step=1.0)
        patrimonio = st.number_input("Total Patrimonio", min_value=0.0, value=10.0, step=1.0)
        
        st.markdown("---")
        
        lista_sectores = ['AGROPECUARIO', 'COMERCIO', 'CONSTRUCCIÓN', 'MANUFACTURA', 'MINERO', 'SERVICIOS']
        sector = st.selectbox("Macrosector", options=lista_sectores)
        
        lista_regiones = ['Antioquia', 'Bogotá - Cundinamarca', 'Centro - Oriente', 'Costa Atlántica', 'Costa Pacífica', 'Eje Cafetero', 'Otros']
        region = st.selectbox("Región", options=lista_regiones)

        st.markdown("<br>", unsafe_allow_html=True)
        # Nota: Algunos botones aún usan use_container_width en versiones viejas, 
        # pero si te da error aquí también, cámbialo a width="stretch" o quita el parámetro.
        boton_calcular = st.form_submit_button("Calcular y Analizar 🚀", type="primary", use_container_width=True)

# --- LÓGICA DE FILTRADO ---
df_filtrado = df_historico[
    (df_historico['MACROSECTOR'] == sector) & 
    (df_historico['REGIÓN'] == region)
].copy()

if not mostrar_ecopetrol:
    df_filtrado = df_filtrado[~df_filtrado['RAZÓN SOCIAL'].str.contains('ECOPETROL', case=False, na=False)]

# --- INTERFAZ PRINCIPAL ---
st.title("📊 Inteligencia Financiera Empresarial")
st.markdown(f"Análisis y proyecciones para el sector **{sector}** en **{region}**.")

tab1, tab2 = st.tabs(["🔮 Modelo Predictivo", "📈 Análisis de Mercado"])

# === PESTAÑA 1: PREDICCIÓN ===
with tab1:
    st.markdown("### Proyección de Ganancias")
    st.markdown("---")
    
    if boton_calcular:
        st.toast("Cálculo completado exitosamente", icon="✅")
        
        datos_entrada = pd.DataFrame({
            'INGRESOS OPERACIONALES': [ingresos], 'TOTAL ACTIVOS': [activos],
            'TOTAL PASIVOS': [pasivos], 'TOTAL PATRIMONIO': [patrimonio],
            'MACROSECTOR': [sector], 'REGIÓN': [region]
        })
        
        try:
            prediccion = modelo.predict(datos_entrada)[0]
            
            col_in, col_res, col_info = st.columns([1, 1.5, 1])
            
            with col_in:
                st.caption("Resumen de Entrada")
                st.write(f"**Ingresos:** ${ingresos:,.1f} MM")
                st.write(f"**Activos:** ${activos:,.1f} MM")
            
            with col_res:
                if prediccion > 0:
                    st.success(f"### Ganancia: ${prediccion:,.2f} MM")
                else:
                    st.error(f"### Pérdida: ${prediccion:,.2f} MM")
            
            with col_info:
                with st.container(border=True):
                    st.metric("Confianza del Modelo (R²)", "61%")
                    st.caption("Explicación de variabilidad basada en datos históricos.")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👈 Ingresa los datos en el menú lateral y presiona 'Calcular'.")

# === PESTAÑA 2: DASHBOARD ===
with tab2:
    st.markdown("### Contexto de Mercado (Datos Reales)")
    
    if df_filtrado.empty:
        st.warning(f"No hay datos para {sector} en {region}.")
    else:
        anio_max = int(df_filtrado['Año de Corte'].max())
        anio_min = int(df_filtrado['Año de Corte'].min())
        
        df_ultimo_anio = df_filtrado[df_filtrado['Año de Corte'] == anio_max]
        
        # CÁLCULOS
        promedio_ingresos_mm = df_ultimo_anio['INGRESOS OPERACIONALES'].mean() * 1000
        tu_ingreso_mm = ingresos * 1000
        
        st.markdown(f"""
        > 📊 **Referencia:** Estás comparando contra el **Promedio del Sector** (${promedio_ingresos_mm:,.0f} Millones).
        > *Nota: Los valores se muestran en Millones de Pesos para facilitar la lectura.*
        """)

        # 1. GRÁFICO COMPARATIVO (Barras)
        st.subheader(f"1. Tu Empresa vs. Promedio del Sector ({anio_max})")
        
        if boton_calcular:
            datos_comparativos = pd.DataFrame({
                'Entidad': ['Tu Empresa', 'Promedio del Sector'],
                'Ingresos (Millones)': [tu_ingreso_mm, promedio_ingresos_mm],
                'Color': ['Tu Empresa', 'Mercado']
            })
            
            fig_bar = px.bar(datos_comparativos, x='Ingresos (Millones)', y='Entidad', color='Color', orientation='h',
                             text_auto=',.0f', 
                             title="Posición en el Mercado (Ingresos en Millones)",
                             color_discrete_map={'Tu Empresa': '#00CC96', 'Mercado': '#636EFA'})
            
            fig_bar.update_layout(xaxis_title="Ingresos (Millones de COP)")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Calcula para ver tu posición en el gráfico.")

        # 2. TENDENCIA (OCUPA TODO EL ANCHO AHORA)
        st.markdown("---")
        st.subheader(f"2. Tendencia del Sector ({anio_min}-{anio_max})")
            
        tendencia = df_filtrado.groupby('Año de Corte')[['INGRESOS OPERACIONALES']].mean().reset_index()
        tendencia['Ingresos (Millones)'] = tendencia['INGRESOS OPERACIONALES'] * 1000
        
        fig_line = px.line(tendencia, x='Año de Corte', y='Ingresos (Millones)', markers=True,
                            title=f"Evolución Promedio de Ingresos")
        
        fig_line.update_xaxes(type='category') 
        fig_line.update_layout(yaxis_title="Ingresos (Millones de COP)")
        
        st.plotly_chart(fig_line, use_container_width=True)
            
        # 3. TABLA TOP 5 (MOVIDA ABAJO Y CON FIX DE WARNING)
        st.markdown("---")
        st.subheader(f"🏆 Top 5 Líderes ({anio_max})")
        
        tabla_top = df_ultimo_anio[['RAZÓN SOCIAL', 'INGRESOS OPERACIONALES', 'REGIÓN']].sort_values(by='INGRESOS OPERACIONALES', ascending=False).head(5)
        tabla_top['INGRESOS OPERACIONALES'] = tabla_top['INGRESOS OPERACIONALES'] * 1000
        tabla_top.columns = ['Empresa', 'Ingresos (Millones)', 'Región']
        
        # FIX APLICADO: width="stretch" en lugar de use_container_width=True
        st.dataframe(
            tabla_top.style.format({'Ingresos (Millones)': '{:,.0f}'}),
            hide_index=True,
            width="stretch" 
        )

        st.markdown("---")
        st.caption("Fuente de Datos:")
        st.link_button("🔗 Ver Datos Abiertos (Supersociedades)", "https://www.datos.gov.co/Comercio-Industria-y-Turismo/10-000-Empresas-mas-Grandes-del-Pa-s/6cat-2gcs/about_data")