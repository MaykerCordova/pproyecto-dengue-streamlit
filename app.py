# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf
# import os

# # --- Configuración de la Página y Modelo ---
# st.set_page_config(
#     page_title="Predicción de Gravedad del Dengue",
#     page_icon="🦟",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ==============================================================================
# # ===> UMBRAL DE PREDICCIÓN RECALIBRADO <===
# # Este valor DEBE ser el mismo que se usó en el script de entrenamiento final.
# PREDICTION_THRESHOLD = 0.45
# # ==============================================================================


# # --- Título y Descripción ---
# st.title("Predicción Inteligente de Gravedad del Dengue 🦟")
# st.markdown(f"""
# Esta aplicación utiliza una red neuronal para predecir si un caso de dengue evolucionará a **'Con signos de alarma'**.
# El modelo está calibrado para lograr un buen equilibrio en la detección de casos graves (umbral de decisión = {PREDICTION_THRESHOLD}).
# """)

# # --- Carga de Modelos y Artefactos ---
# @st.cache_resource
# def load_artifacts():
#     """Carga todos los componentes del modelo desde el disco."""
#     base_dir = 'modelo_final_dengue'
#     try:
#         model = tf.keras.models.load_model(os.path.join(base_dir, 'dengue_model.h5'))
#         preprocessor = joblib.load(os.path.join(base_dir, 'preprocessor.pkl'))
#         label_encoder = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
#         model_columns = joblib.load(os.path.join(base_dir, 'model_columns.pkl'))
#         return model, preprocessor, label_encoder, model_columns
#     except (FileNotFoundError, IOError) as e:
#         st.error(f"Error al cargar los archivos del modelo: {e}. Asegúrese de haber ejecutado el script 'entrenamiento_completo_final.py' primero.")
#         return None, None, None, None

# model, preprocessor, label_encoder, model_columns = load_artifacts()

# if model is None:
#     st.stop()

# # --- Barra Lateral para Entradas del Usuario ---
# st.sidebar.header("Ingrese los Datos del Paciente")

# def user_input_features():
#     """Crea los widgets para la entrada de datos y devuelve un DataFrame."""
#     sexo = st.sidebar.selectbox('Sexo', ['M', 'F', 'I'])
#     # Se eliminaron 'estado' y 'area' porque no forman parte del modelo final.
    
#     municipio_procedencia = st.sidebar.selectbox('Municipio de Procedencia', ['Cali', 'Buenaventura', 'Palmira', 'Otro'])
    
#     st.sidebar.markdown("---")
#     st.sidebar.subheader("Síntomas y Datos Clínicos")
    
#     fiebre = st.sidebar.selectbox('¿Presenta Fiebre?', ['SI', 'NO'])
#     cefalea = st.sidebar.selectbox('¿Presenta Cefalea (dolor de cabeza)?', ['SI', 'NO'])
#     malgias = st.sidebar.selectbox('¿Presenta Malgias (dolor muscular)?', ['SI', 'NO'])
#     artralgia = st.sidebar.selectbox('¿Presenta Artralgia (dolor articular)?', ['SI', 'NO'])
#     erupcion = st.sidebar.selectbox('¿Presenta Erupción Cutánea?', ['SI', 'NO'])
    
#     st.sidebar.markdown("---")
    
#     edad = st.sidebar.number_input('Edad (años)', 0, 120, 25, 1)
#     dias_hasta_consulta = st.sidebar.number_input('Días desde inicio de síntomas', 0, 30, 3)
#     mes_sintomas = st.sidebar.slider('Mes de inicio de síntomas', 1, 12, 6)

#     data = {
#         'edad': edad,
#         'sexo': sexo,
#         'nombre_municipio_procedencia': municipio_procedencia,
#         'fiebre': fiebre,
#         'cefalea': cefalea,
#         'malgias': malgias,
#         'artralgia': artralgia,
#         'erupcion': erupcion,
#         'dias_hasta_consulta': dias_hasta_consulta,
#         'mes_sintomas': mes_sintomas
#     }
    
#     features = pd.DataFrame(data, index=[0])
#     # Asegura que el DataFrame tenga las columnas en el orden exacto que el modelo espera
#     return features[model_columns]

# input_df = user_input_features()

# # --- Visualización de Datos de Entrada ---
# st.subheader('Resumen de Datos Ingresados')
# st.dataframe(input_df.T.rename(columns={0: 'Valores'}))

# # --- Botón de Predicción y Resultados ---
# if st.sidebar.button('Predecir Gravedad', type="primary"):
#     input_processed = preprocessor.transform(input_df)
    
#     # La salida del modelo es la probabilidad de la clase 1 ('Sin signos de alarma').
#     prob_clase_no_critica = model.predict(input_processed)[0][0]
#     prob_clase_critica = 1 - prob_clase_no_critica
    
#     # Decidimos la clase final usando el UMBRAL RECALIBRADO.
#     if prob_clase_critica >= PREDICTION_THRESHOLD:
#          prediction_class_idx = 0 # 'Con signos de alarma'
#     else:
#          prediction_class_idx = 1 # 'Sin signos de alarma'

#     prediction_class_name = label_encoder.inverse_transform([prediction_class_idx])[0]

#     # --- Mostrar Resultados ---
#     st.subheader('Resultado de la Predicción')
    
#     col1, col2 = st.columns(2)
#     with col1:
#         if prediction_class_name == 'Con signos de alarma':
#             st.error(f"**Diagnóstico Predicho: {prediction_class_name}**")
#             st.warning("Se recomienda monitoreo cercano y evaluación médica detallada.")
#         else:
#             st.success(f"**Diagnóstico Predicho: {prediction_class_name}**")
#             st.info("El riesgo de complicaciones parece bajo. Seguir indicaciones médicas.")
#     with col2:
#         st.metric("Probabilidad de desarrollar signos de alarma", f"{prob_clase_critica:.2%}")
#         st.progress(prob_clase_critica)

#     st.expander("¿Cómo interpretar este resultado?").info(f"""
#     La red neuronal ha calculado una probabilidad del **{prob_clase_critica:.2%}** de que el paciente desarrolle signos de alarma.
#     Debido a que esta probabilidad es {'mayor o igual' if prob_clase_critica >= PREDICTION_THRESHOLD else 'menor'} que nuestro umbral de decisión de **{PREDICTION_THRESHOLD:.2%}**, 
#     el modelo clasifica el caso como **'{prediction_class_name}'**.
#     """)
# else:
#     st.info('Ingrese los datos en la barra lateral y haga clic en "Predecir" para obtener un diagnóstico.')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

st.set_page_config(
    page_title="Predicción de Gravedad del Dengue",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

PREDICTION_THRESHOLD = 0.45

st.title("Predicción Inteligente de Gravedad del Dengue 🦟")
st.markdown(f"Esta aplicación utiliza una red neuronal para predecir si un caso de dengue evolucionará a **'Con signos de alarma'**.")

@st.cache_resource
def load_artifacts():
    base_dir = 'modelo_final_dengue'
    try:
        model = tf.keras.models.load_model(os.path.join(base_dir, 'dengue_model.h5'))
        preprocessor = joblib.load(os.path.join(base_dir, 'preprocessor.pkl'))
        label_encoder = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
        model_columns = joblib.load(os.path.join(base_dir, 'model_columns.pkl'))
        return model, preprocessor, label_encoder, model_columns
    except (FileNotFoundError, IOError) as e:
        st.error(f"Error al cargar archivos del modelo: {e}. Ejecute el script de entrenamiento.")
        return None, None, None, None

model, preprocessor, label_encoder, model_columns = load_artifacts()

if model is None:
    st.stop()

st.sidebar.header("Ingrese los Datos del Paciente")

def user_input_features():
    # >> CAMBIO 1: Las opciones para Sexo ahora solo son 'M' y 'F'
    sexo = st.sidebar.selectbox('Sexo', ['M', 'F'])
    
    # >> CAMBIO 2: Se eliminó el selectbox para 'municipio_procedencia'
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Síntomas y Datos Clínicos")
    
    fiebre = st.sidebar.selectbox('¿Presenta Fiebre?', ['SI', 'NO'])
    cefalea = st.sidebar.selectbox('¿Presenta Cefalea?', ['SI', 'NO'])
    malgias = st.sidebar.selectbox('¿Presenta Malgias?', ['SI', 'NO'])
    artralgia = st.sidebar.selectbox('¿Presenta Artralgia?', ['SI', 'NO'])
    erupcion = st.sidebar.selectbox('¿Presenta Erupción Cutánea?', ['SI', 'NO'])
    
    st.sidebar.markdown("---")
    
    edad = st.sidebar.number_input('Edad (años)', 0, 120, 25, 1)
    dias_hasta_consulta = st.sidebar.number_input('Días desde inicio de síntomas', 0, 30, 3)
    mes_sintomas = st.sidebar.slider('Mes de inicio de síntomas', 1, 12, 6)

    # >> CAMBIO 3: El diccionario de datos ya no incluye 'nombre_municipio_procedencia'
    data = {
        'edad': edad,
        'sexo': sexo,
        'fiebre': fiebre,
        'cefalea': cefalea,
        'malgias': malgias,
        'artralgia': artralgia,
        'erupcion': erupcion,
        'dias_hasta_consulta': dias_hasta_consulta,
        'mes_sintomas': mes_sintomas
    }
    
    features = pd.DataFrame(data, index=[0])
    return features[model_columns]

input_df = user_input_features()

st.subheader('Resumen de Datos Ingresados')
st.dataframe(input_df.T.rename(columns={0: 'Valores'}))

if st.sidebar.button('Predecir Gravedad', type="primary"):
    input_processed = preprocessor.transform(input_df)
    prob_clase_no_critica = model.predict(input_processed)[0][0]
    prob_clase_critica = 1 - prob_clase_no_critica
    
    if prob_clase_critica >= PREDICTION_THRESHOLD:
         prediction_class_idx = 0
    else:
         prediction_class_idx = 1

    prediction_class_name = label_encoder.inverse_transform([prediction_class_idx])[0]

    st.subheader('Resultado de la Predicción')
    col1, col2 = st.columns(2)
    with col1:
        if prediction_class_name == 'Con signos de alarma':
            st.error(f"**Diagnóstico Predicho: {prediction_class_name}**")
            st.warning("Se recomienda monitoreo cercano.")
        else:
            st.success(f"**Diagnóstico Predicho: {prediction_class_name}**")
            st.info("El riesgo de complicaciones parece bajo.")
    with col2:
        st.metric("Probabilidad de desarrollar signos de alarma", f"{prob_clase_critica:.2%}")
        st.progress(prob_clase_critica)