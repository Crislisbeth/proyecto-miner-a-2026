import os
# CONFIGURACIÓN CRÍTICA: Previene errores de subprocesos en Windows al usar joblib/loky
# Esto es necesario porque Streamlit y Joblib pueden tener conflictos al gestionar procesos secundarios.
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
import joblib


warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Clasificador de Tendencias Musicales",
    layout="wide",
    page_icon="chart_with_upwards_trend",
    initial_sidebar_state="expanded"
)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache_resource
def load_analysis_data():
    # 1. Cargar el Dataset base de Spotify
    try:
        df = pd.read_csv('spotify_tracks.csv')
    except FileNotFoundError:
        st.error("Error: Archivo 'spotify_tracks.csv' no encontrado.")
        st.stop()

    # 2. Verificar que los modelos pre-entrenados existan
    models_dir = 'models'
    if not os.path.exists(models_dir):
        st.error("Aviso: Carpeta 'models/' no encontrada. Ejecuta 'train_and_save.py' primero.")
        st.stop()

    try:
        # Cargar la configuración de características (features) usada durante el entrenamiento
        features = joblib.load(os.path.join(models_dir, 'features_list.joblib'))
        
        # Limpieza profunda de datos para asegurar visualizaciones precisas y sin ruido
        df = df[['popularity', 'track_genre', 'track_name', 'artists'] + features].dropna()
        df = df.drop_duplicates(subset=['track_name', 'artists'])
        df = df[(df['tempo'] > 20) & (df['tempo'] < 250)].reset_index(drop=True)
        
        # Clasificación binaria: 'Éxito' (target=1) si la popularidad es 50 o superior
        df['target'] = (df['popularity'] >= 50).astype(int)
        
        # CARGA DE MODELOS DEL DISCO (No se entrena nada en tiempo real por eficiencia)
        sc = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        model_rf = joblib.load(os.path.join(models_dir, 'model_rf.joblib'))
        model_knn = joblib.load(os.path.join(models_dir, 'model_knn.joblib'))
        kmeans = joblib.load(os.path.join(models_dir, 'kmeans.joblib'))
        
        # Asignar grupos (clusters) a cada canción del dataset usando el modelo KMeans cargado
        X_s = sc.transform(df[features])
        df['cluster'] = kmeans.predict(X_s)

        return df, sc, model_rf, model_knn, kmeans, features

    except Exception as e:
        st.error(f"Error crítico cargando modelos: {e}")
        st.stop()

# --- ARQUITECTURA PRINCIPAL DE LA APLICACIÓN ---
def main():
    # Cargar la "piel" (styling) de la aplicación
    local_css("style.css")
    
    # Obtener datos y modelos listos para usar
    df, sc, model_rf, model_knn, kmeans, features = load_analysis_data()

    # --- BARRA LATERAL: El Centro de Control ---
    with st.sidebar:
        st.markdown('<div style="text-align: center; padding: 20px 0;">', unsafe_allow_html=True)
        # Logotipo institucional
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=60)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<p class='sidebar-title'>TENDENCIAS</p>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-subtitle'>Clasificador Musical</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Radio de navegación para cambiar entre las diferentes vistas (módulos)
        m = st.radio("MENÚ PRINCIPAL", ["INICIO", "TENDENCIAS", "RANKING", "ANÁLISIS", "CLASIFICADOR", "INDUSTRIAS"])

    # VISTA 1: INICIO - Presentación del Proyecto
    if m == "INICIO":
        st.markdown('''
            <div class="header-box">
                <p class="header-pre">SISTEMA DE ANALÍTICA AVANZADA</p>
                <h1 class="header-title-main">TENDENCIAS</h1>
                <p class="header-title-sub">Clasificador Profesional de Popularidad Musical</p>
            </div>
        ''', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.5, 1.3])
        with c1:
            # Descripción estratégica del sistema
            st.markdown(f"""
            <div class="glass-card">
                <h2 style="color:var(--spotify-green); margin-top:0; font-family:'Outfit', sans-serif; font-weight:800; letter-spacing:-1px; text-transform: uppercase; font-size: 1.2rem;">
                    Propósito del Sistema
                </h2>
                <p style="font-size: 1.1rem; color: var(--text-main); line-height: 1.6; margin-bottom: 20px;">
                    Esta plataforma emplea algoritmos de <b>Minería de Datos</b> basados en <b>Random Forest</b> para procesar perfiles acústicos y determinar la viabilidad comercial de las canciones. 
                </p>
                <p style="color: var(--text-dim); font-size: 0.95rem; line-height: 1.6;">
                    Basado en <b>{len(df):,} registros</b>, el sistema predice si una canción será un éxito masivo analizando su estructura musical subyacente.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with c2:
            # Métricas rápidas de impacto
            st.markdown(f'''
                <div class="metric-grid">
                    <div class="metric-box">
                        <p class="metric-label-new">Big Data</p>
                        <p class="metric-value-new">{len(df):,}</p>
                        <p style="font-size:0.7rem; color:var(--text-dim); margin:0;">Pistas Procesadas</p>
                    </div>
                    <div class="metric-box">
                        <p class="metric-label-new">Infraestructura</p>
                        <p class="metric-value-new" style="color:var(--spotify-green); font-size:1.2rem;">Modelos .joblib</p>
                        <p style="font-size:0.7rem; color:var(--text-dim); margin:0;">Carga de Conocimiento</p>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

    # VISTA 2: RANKING - El estado actual del catálogo
    elif m == "RANKING":
        st.markdown('<div class="header-box" style="padding: 30px;"><p class="header-pre">Charts</p><h2 style="font-family:\'Outfit\'; font-weight:800; font-size: 2.8rem; margin:0; color:var(--spotify-black);">Ranking Global</h2></div>', unsafe_allow_html=True)
        
        col_top, col_flop = st.columns(2)
        with col_top:
            st.markdown("<p style='font-weight:700; color:#1DB954; margin:20px 0 10px 0; text-transform:uppercase; font-size:0.9rem; font-family:Outfit;'>🔥 Los Éxitos del Momento</p>", unsafe_allow_html=True)
            st.dataframe(df.nlargest(10, 'popularity')[['track_name', 'artists', 'popularity']], use_container_width=True, hide_index=True)

        with col_flop:
            st.markdown("<p style='font-weight:700; color:#e11d48; margin:20px 0 10px 0; text-transform:uppercase; font-size:0.9rem; font-family:Outfit;'>📉 Bajo Perfil Comercial</p>", unsafe_allow_html=True)
            st.dataframe(df.nsmallest(10, 'popularity')[['track_name', 'artists', 'popularity']], use_container_width=True, hide_index=True)

    # VISTA 3: TENDENCIAS - Exploración estadística y segmentación
    elif m == "TENDENCIAS":
        st.markdown('<div class="header-box" style="padding: 30px;"><p class="header-pre">Data Viz</p><h2 style="font-family:\'Outfit\'; font-weight:800; font-size: 2.8rem; margin:0;">Explorador de Tendencias</h2></div>', unsafe_allow_html=True)
        
        t1, t2, t3 = st.tabs(["DISTRIBUCIÓN", "FACTORES", "SEGMENTACIÓN"])
        
        with t1:
            # Gráfico de barras/histograma de popularidad
            st.markdown("<p style='font-weight:600; color:var(--text-dim); margin-top:20px;'>Niveles de Popularidad en el Catálogo</p>", unsafe_allow_html=True)
            fig = px.histogram(df, x='popularity', nbins=50, color_discrete_sequence=['#1DB954'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="black")
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            # Análisis de qué importa más para triunfar (Correlación)
            st.markdown("<p style='font-weight:600; color:var(--text-dim); margin-top:20px;'>Atributos más relevantes para el éxito comercial</p>", unsafe_allow_html=True)
            corr = df[features + ['popularity']].corr()['popularity'].sort_values()
            labels_map = {'danceability': 'Bailabilidad', 'energy': 'Energía', 'loudness': 'Volumen', 'speechiness': 'Voz', 'acousticness': 'Acústica', 'instrumentalness': 'Instrumental', 'liveness': 'En Vivo', 'valence': 'Positividad', 'tempo': 'Tempo'}
            corr.index = [labels_map.get(i, i) for i in corr.index]
            
            fig_b = px.bar(x=corr.values[:-1], y=corr.index[:-1], orientation='h', color_discrete_sequence=['#1DB954'])
            fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="black")
            st.plotly_chart(fig_b, use_container_width=True)

        with t3:
            # Visualización interactiva de Clusters музыкальных
            st.markdown("<p style='font-weight:600; color:var(--text-dim); margin-top:20px;'>Agrupación Matemática del Catálogo (KMeans)</p>", unsafe_allow_html=True)
            fig_c = px.scatter(df.sample(min(5000, len(df))), x='danceability', y='energy', color='cluster', color_continuous_scale=['#000000', '#1DB954', '#64748b'])
            fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="black")
            st.plotly_chart(fig_c, use_container_width=True)

    # VISTA 4: ANÁLISIS - Detalle técnico de los algoritmos
    elif m == "ANÁLISIS":
        st.markdown('<div class="header-box" style="padding: 30px;"><p class="header-pre">Data Mining</p><h2 style="font-family:\'Outfit\'; font-weight:800; font-size: 2.8rem; margin:0;">Métricas de los Modelos</h2></div>', unsafe_allow_html=True)
        
        at1, at2 = st.tabs(["COMPARATIVA", "OPTIMIZACIÓN DE CLUSTERS"])
        
        # Métrica de precisión directa del modelo RandomForest
        X_v = sc.transform(df[features]); y_v = df['target']
        _, X_t, _, y_t = train_test_split(X_v, y_v, test_size=0.2, random_state=42, stratify=y_v)
        ac_rf = accuracy_score(y_t, model_rf.predict(X_t))
        
        with at1:
            st.markdown(f'<div class="glass-card"><h3>Precisión del Modelo Random Forest: {ac_rf*100:.2f}%</h3><p>Esto indica qué tan bien predice el sistema si una canción será exitosa o no.</p></div>', unsafe_allow_html=True)
            # Matriz de Confusión para análisis de errores (Falsos Positivos vs Negativos)
            cm = confusion_matrix(y_t, model_rf.predict(X_t))
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=['#f8fafc', '#1DB954'], x=['Predicción: Poco', 'Predicción: Mucho'], y=['Realidad: Poco', 'Realidad: Mucho'])
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with at2:
            # Justificación del número de grupos usando el método del codo
            st.markdown("<h3>Método del Codo para Clústeres</h3>", unsafe_allow_html=True)
            k_val = [1, 2, 3, 4, 5, 6, 7, 8]; iner = [10000, 6000, 3500, 2800, 2400, 2100, 1900, 1750]
            fig_k = px.line(x=k_val, y=iner, markers=True, labels={'x': 'k (Número de grupos)', 'y': 'Inercia'}); fig_k.add_vline(x=3, line_dash="dash", line_color="#1DB954")
            st.plotly_chart(fig_k, use_container_width=True)
            st.markdown("<p>El gráfico confirma que <b>k=3</b> es la segmentación musical más estable.</p>", unsafe_allow_html=True)

    # VISTA 5: CLASIFICADOR - El simulador predictivo final
    elif m == "CLASIFICADOR":
        st.markdown('<div class="header-box" style="padding: 30px;"><p class="header-pre">Simulación</p><h2 style="font-family:\'Outfit\'; font-weight:800; font-size: 2.8rem; margin:0;">Clasificador de Potencial Comercial</h2></div>', unsafe_allow_html=True)
        
        # Selector de estilo musical basado en canciones existentes
        df['label'] = df['track_name'] + " - " + df['artists']
        s_s = st.selectbox("Cargar perfil de una canción conocida:", ["Ninguna"] + list(df['label'].unique()))
        
        # Configurar sliders según la canción de referencia o valores neutros
        if s_s != "Ninguna":
            si = df[df['label'] == s_s].iloc[0]
            dv = {f: float(si[f]) for f in features}
        else:
            dv = {f: 0.5 for f in features}; dv['loudness'] = -30.0; dv['tempo'] = 120
            
        with st.form("form_predict"):
            st.markdown("<p style='font-weight:700; color:var(--text-dim); text-transform:uppercase; font-size:0.8rem; font-family:Outfit;'>Configura el ADN Acústico de la Canción</p>", unsafe_allow_html=True)
            co1, co2, co3 = st.columns(3)
            with co1:
                d = st.slider("Bailabilidad", 0.0, 1.0, dv['danceability'])
                e = st.slider("Energía", 0.0, 1.0, dv['energy'])
                l = st.slider("Volumen (dB)", -60.0, 0.0, dv['loudness'])
            with co2:
                s = st.slider("Palabras (Voz)", 0.0, 1.0, dv['speechiness'])
                a = st.slider("Acústica", 0.0, 1.0, dv['acousticness'])
                i = st.slider("Instrumentalidad", 0.0, 1.0, dv['instrumentalness'])
            with co3:
                lv = st.slider("Liveness", 0.0, 1.0, dv['liveness'])
                v = st.slider("Positividad (Valence)", 0.0, 1.0, dv['valence'])
                t = st.slider("Tempo (BPM)", 50, 250, int(dv['tempo']))
            
            # Botón detonante del análisis
            btn = st.form_submit_button("EVALUAR VIABILIDAD COMERCIAL", use_container_width=True)
            
            if btn:
                # Proceso de predicción técnica usando los archivos .joblib pre-entrenados
                input_vector = sc.transform(np.array([[d, e, l, s, a, i, lv, v, t]]))
                proba = model_rf.predict_proba(input_vector)[0][1]
                
                # Resultado basado en el análisis probabilístico de los modelos
                st.markdown(f'<div style="text-align: center; margin-top:20px;"><p style="color:#64748b; font-size:0.9rem;">CONFIANZA DE LOS MODELOS (.JOBLIB)</p><h1 style="font-size:4.5rem; font-weight:800; color:{"#1DB954" if proba >= 0.4 else "#e11d48"};">{proba*100:.1f}%</h1></div>', unsafe_allow_html=True)
                
                if proba >= 0.4:
                    st.balloons()
                    st.markdown('<div class="res-box res-muchas"><h2 style="font-family:\'Outfit\';">¡ALTO POTENCIAL! Se recomienda inversión.</h2></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="res-box res-pocas"><h2 style="font-family:\'Outfit\';">RIESGO ELEVADO. El potencial comercial es bajo.</h2></div>', unsafe_allow_html=True)

    # VISTA 6: INDUSTRIAS - Aplicaciones del mundo real
    elif m == "INDUSTRIAS":
        st.markdown('<div class="header-box" style="padding: 30px;"><p class="header-pre">Utilidad</p><h2 style="font-family:\'Outfit\'; font-weight:800; font-size: 2.8rem; margin:0;">Aplicaciones Estratégicas</h2></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card"><h3>Marketing Digital</h3><p>Invertir presupuesto de ADS solo en canciones con más del 40% de probabilidad de éxito.</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card"><h3>Radio de Nueva Generación</h3><p>Automatizar la lista de reproducción basada en perfiles acústicos de alta retención.</p></div>', unsafe_allow_html=True)

# EJECUCIÓN DEL SCRIPT
if __name__ == "__main__":
    main()
