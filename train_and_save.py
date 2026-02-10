import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# Configuración para Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

print("Iniciando proceso de entrenamiento y guardado de modelos...")

# 1. Carga de datos
try:
    df = pd.read_csv('spotify_tracks.csv')
    print(f"Datos cargados: {len(df)} registros.")
except FileNotFoundError:
    print("Error: No se encuentra spotify_tracks.csv")
    exit()

# 2. Preprocesamiento
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
df_clean = df[['popularity', 'track_genre', 'track_name', 'artists'] + features].dropna()
df_clean = df_clean.drop_duplicates(subset=['track_name', 'artists'])
df_clean = df_clean[(df_clean['tempo'] > 20) & (df_clean['tempo'] < 250)].reset_index(drop=True)

# 3. Transformación
df_clean['target'] = (df_clean['popularity'] >= 50).astype(int)
X = df_clean[features]
y = df_clean['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Minería (Entrenamiento)
print("Entrenando RandomForest (esto puede tardar unos segundos)...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train, y_train)

print("Entrenando KNN...")
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)

print("Entrenando KMeans...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 5. Persistencia (Guardado)
if not os.path.exists('models'):
    os.makedirs('models')

print("Guardando modelos en carpeta 'models/'...")
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(rf, 'models/model_rf.joblib')
joblib.dump(knn, 'models/model_knn.joblib')
joblib.dump(kmeans, 'models/kmeans.joblib')
joblib.dump(features, 'models/features_list.joblib')

print("¡PROCESO COMPLETADO! Modelos listos para usar en app.py")
