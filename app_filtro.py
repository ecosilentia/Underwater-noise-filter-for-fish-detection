import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import io

# Cargar modelo una sola vez
@st.cache_resource
def cargar_modelo(ruta_modelo):
    modelo = load_model(ruta_modelo)
    return modelo

# Función para procesar audio
def procesar_audio(file_buffer, sr, n_fft, hop_length, modelo, porcentaje_filtro, max_duracion=60):
    # Cargar audio
    audio, sr_real = librosa.load(file_buffer, sr=sr)

    # Recortar si el audio es más largo que max_duracion segundos
    if librosa.get_duration(y=audio, sr=sr_real) > max_duracion:
        audio = audio[:max_duracion * sr_real]
        recortado = True
    else:
        recortado = False

    # Espectrograma
    S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Normalizar
    S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    img_espectro = (S_db_norm * 255).astype(np.uint8)

    # Redimensionar para predicción
    img_gray = cv2.resize(img_espectro, (387, 385))
    input_modelo = img_gray.reshape(1, 385, 387, 1) / 255.0

    # Predicción
    pred = modelo.predict(input_modelo)[0][0]

    # Aplicar filtro controlado
    if pred >= 0.5:
        umbral = -35 + ((-80 + 35) * (porcentaje_filtro / 10))
        S_db_filtrado = np.where(S_db > umbral, -80, S_db)
    else:
        S_db_filtrado = S_db

    return pred, S_db, S_db_filtrado, recortado

# Función para reducir espectrograma dinámicamente
def reducir_espectrograma(S_db, max_cols=1000):
    if S_db.shape[1] > max_cols:
        factor = S_db.shape[1] // max_cols
        return S_db[:, ::factor], factor
    else:
        return S_db, 1

# -------------------------
# INICIO DE LA APP
# -------------------------

st.title("🎛️ Filtro Inteligente de Ruido Submarino")
st.write("Sube un archivo de audio y ajusta el porcentaje de filtro.")

# Parámetros
sr = 6000
n_fft = 5000
hop_length = 40
duracion_maxima = 30  # segundos

# Cargar modelo
ruta_modelo = "modelo_filtro_ruido.keras"  # Ajusta esta ruta si es necesario
modelo = cargar_modelo(ruta_modelo)

# Slider para porcentaje de filtro
porcentaje_filtro = st.slider('🔧 Porcentaje de filtro', 0, 10, 5)

# Subir archivo
archivo_audio = st.file_uploader("Sube tu archivo WAV", type=["wav"])

if archivo_audio is not None:
    pred, S_db, S_db_filtrado, recortado = procesar_audio(archivo_audio, sr, n_fft, hop_length, modelo, porcentaje_filtro, max_duracion=duracion_maxima)

    st.write(f"🎯 Predicción de Ruido (0=silencio, 1=ruido): **{pred:.3f}**")
    
    if recortado:
        st.warning(f"⚠️ Audio recortado a los primeros {duracion_maxima} segundos para evitar problemas de memoria.")

    # 🛠️ REDUCIR dinámicamente el tamaño del espectrograma para graficar
    S_db_reducido, factor = reducir_espectrograma(S_db)
    S_db_filtrado_reducido, _ = reducir_espectrograma(S_db_filtrado)

    # Graficar
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    librosa.display.specshow(S_db_reducido, sr=sr, hop_length=hop_length*factor,
                             x_axis='time', y_axis='log', ax=ax[0], cmap='gray_r')
    ax[0].set_title('🎧 Espectrograma Original')
    ax[0].set_ylim(140, 2300)

    librosa.display.specshow(S_db_filtrado_reducido, sr=sr, hop_length=hop_length*factor,
                             x_axis='time', y_axis='log', ax=ax[1], cmap='gray_r')
    ax[1].set_title('🔇 Espectrograma Filtrado')
    ax[1].set_ylim(140, 2300)

    plt.tight_layout()
    st.pyplot(fig)

