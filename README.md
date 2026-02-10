# Clasificador de Tendencias Musicales - Proyecto de Minería de Datos

Este proyecto consiste en una plataforma avanzada de analítica y predicción de popularidad musical, desarrollada como parte de la asignatura de **Minería de Datos**. El sistema utiliza modelos de aprendizaje supervisado y no supervisado para analizar perfiles acústicos y determinar la viabilidad comercial de las obras musicales.

---

## 🚀 Propósito del Proyecto
El objetivo principal es identificar qué atributos acústicos (como energía, bailabilidad o volumen) influyen en que una canción supere el umbral de las **50 unidades de popularidad** en Spotify. El sistema permite:
- **Visualizar tendencias** globales del catálogo.
- **Segmentar canciones** en grupos basados en similitud musical.
- **Predecir el éxito** de nuevas composiciones antes de su lanzamiento.

---

## 🧠 Arquitectura de los Modelos (.joblib)
A diferencia de otros sistemas, esta aplicación consume directamente **conocimiento pre-entrenado** almacenado en la carpeta `models/`. Los archivos clave son:

- **`model_rf.joblib`**: El "cerebro" principal basado en **Random Forest**. Analiza múltiples variables para dar una probabilidad de éxito comercial.
- **`model_knn.joblib`**: Modelo de **K-Nearest Neighbors** usado para comparar la eficiencia predictiva.
- **`kmeans.joblib`**: Algoritmo de **Clustering** que segmenta el catálogo en 3 grandes nichos musicales.
- **`scaler.joblib`**: Objeto de normalización que asegura que los datos de entrada estén en la misma escala que los de entrenamiento.
- **`features_list.joblib`**: Define el orden exacto de las variables acústicas para el procesamiento.

---

## 🛠 Estructura del Proyecto
- `app.py`: La interfaz interactiva profesional desarrollada en Streamlit.
- `train_and_save.py`: Script encargado de procesar el dataset original (`spotify_tracks.csv`) y generar los modelos `.joblib`.
- `style.css`: Estilos visuales premium inspirados en la estética de Spotify.
- `models/`: Directorio que contiene los modelos persistidos.

---

## 📋 ¿Cómo utilizarlo?

### 1. Preparar el Entorno
Asegúrate de tener instaladas las dependencias necesarias:
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

### 2. Generar los Modelos (Opcional si ya existen)
Si los archivos en `models/` no están presentes, ejecuta el script de entrenamiento:
```bash
python train_and_save.py
```

### 3. Ejecutar la Aplicación
Lanza la plataforma interactiva con el siguiente comando:
```bash
streamlit run app.py
```

---

## 📊 Secciones de la Plataforma
1. **Inicio:** Introducción técnica y métricas generales del dataset.
2. **Tendencias:** Histogramas de alcance, análisis de correlación y visualización de clústeres.
3. **Ranking:** Top 10 de canciones más y menos populares del catálogo.
4. **Análisis:** Evaluación técnica de la precisión de los modelos y el método del codo.
5. **Clasificador:** Laboratorio predictivo donde puedes configurar el "ADN" de una canción y ver su probabilidad de éxito.

---
**Nota:** Este sistema es una herramienta de apoyo basada en minería de datos y debe usarse como referencia probabilística, no como una garantía absoluta de mercado.
