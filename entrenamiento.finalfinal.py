# # ==============================================================================
# # SCRIPT DE ENTRENAMIENTO FINAL, COMPLETO Y RECALIBRADO
# #
# # VERSIÓN DEFINITIVA QUE INCLUYE:
# # - Ajustes en class_weight y umbral para Keras para evitar el colapso del modelo.
# # - Corrección de nombres de columnas.
# # - Lógica correcta para predicción de alto recall.
# # - Tuneo avanzado para Scikit-learn.
# # ==============================================================================

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import os
# import random

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.neural_network import MLPClassifier
# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.over_sampling import SMOTE

# # --- CONFIGURACIÓN GLOBAL RECALIBRADA ---
# DATA_PATH = 'Dengue.csv'
# MODEL_OUTPUT_DIR = 'modelo_final_dengue'
# TARGET = 'clasificacion_final'
# RANDOM_STATE = 42
# # UMBRAL DE DECISIÓN AJUSTADO PARA UN MEJOR EQUILIBRIO
# KERAS_PREDICTION_THRESHOLD = 0.46

# # --- FIJAR SEMILLAS PARA REPRODUCIBILIDAD ---
# os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
# random.seed(RANDOM_STATE)
# np.random.seed(RANDOM_STATE)
# tf.random.set_seed(RANDOM_STATE)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

# # Crear directorio de salida
# os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# def cargar_y_preparar_datos(path):
#     """
#     Carga los datos, realiza ingeniería de características y selecciona
#     explícitamente las columnas finales del modelo.
#     """
#     print("--- 1. Cargando y Preparando Datos ---")
#     try:
#         df = pd.read_csv(path, encoding='utf-8')
#     except UnicodeDecodeError:
#         df = pd.read_csv(path, encoding='latin1')
    
#     df.columns = df.columns.str.strip()

#     for col in ['fecha_nototificacion', 'fecha_consulta', 'fecha_inicio_sintomas']:
#         df[col] = pd.to_datetime(df[col], errors='coerce')
    
#     df = df.dropna(subset=['fecha_consulta', 'fecha_inicio_sintomas'])
#     df['dias_hasta_consulta'] = (df['fecha_consulta'] - df['fecha_inicio_sintomas']).dt.days
#     df = df[(df['dias_hasta_consulta'] >= 0) & (df['dias_hasta_consulta'] < 30)]
#     df['mes_sintomas'] = df['fecha_inicio_sintomas'].dt.month

#     df[TARGET] = df[TARGET].replace({
#         'DENGUE GRAVE': 'Con signos de alarma', 
#         'DENGUE CON SIGNOS DE ALARMA': 'Con signos de alarma',
#         'DENGUE SIN SIGNOS DE ALARMA': 'Sin signos de alarma'
#     })
    
#     final_model_columns = [
#         'edad', 'sexo', 'nombre_municipio_procedencia', 'fiebre', 'cefalea', 
#         'malgias', 'artralgia', 'erupcion', 'dias_hasta_consulta', 'mes_sintomas',
#         TARGET
#     ]
    
#     df_final = df.reindex(columns=final_model_columns)
#     df_final = df_final.dropna().reset_index(drop=True)

#     print(f"\n[INFO] Forma del DataFrame final listo para entrenar: {df_final.shape}")
#     if df_final.empty:
#         print("[ERROR CRÍTICO] El DataFrame está vacío. Revisa los nombres en 'final_model_columns'.")
#         exit()

#     for col in df_final.select_dtypes(include='object').columns:
#         if col != TARGET:
#             df_final[col] = df_final[col].astype(str)
            
#     return df_final


# def analisis_exploratorio(df):
#     """Realiza y guarda un análisis exploratorio básico."""
#     print("\n--- 2. Análisis Exploratorio de Datos (EDA) ---")
#     print("Distribución de la variable objetivo:")
#     print(df[TARGET].value_counts(normalize=True))
    
#     plt.figure(figsize=(8, 5))
#     sns.countplot(x=TARGET, data=df, palette='viridis')
#     plt.title('Distribución de Clases (Dengue)')
#     plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'distribucion_clases.png'))
#     plt.close()


# def preprocesar_y_dividir(df):
#     """Define preprocesador, divide datos y los transforma."""
#     print("\n--- 3. Preprocesamiento y División de Datos ---")
    
#     X = df.drop(columns=[TARGET])
#     y_raw = df[TARGET]

#     joblib.dump(X.columns.tolist(), os.path.join(MODEL_OUTPUT_DIR, 'model_columns.pkl'))

#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y_raw)
    
#     numerical_features = X.select_dtypes(include=np.number).columns.tolist()
#     categorical_features = X.select_dtypes(include='object').columns.tolist()
    
#     preprocessor = ColumnTransformer(transformers=[
#         ('num', StandardScaler(), numerical_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
#     ], remainder='passthrough')

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
#     X_train_processed = preprocessor.fit_transform(X_train)
#     X_test_processed = preprocessor.transform(X_test)
    
#     return X_train_processed, X_test_processed, y_train, y_test, preprocessor, label_encoder


# def evaluar_modelo(model_name, y_test, y_pred, label_encoder, history=None):
#     """Evalúa el rendimiento del modelo y genera gráficos."""
#     print(f"\n--- 6. Evaluación del Modelo: {model_name} ---")
    
#     class_names = label_encoder.classes_
#     report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
#     print(report)
    
#     cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.title(f'Matriz de Confusión - {model_name}')
#     plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
#     plt.close()
    
#     if history:
#         pd.DataFrame(history.history).plot(figsize=(10, 6))
#         plt.grid(True)
#         plt.gca().set_ylim(0, 1)
#         plt.title(f'Curvas de Aprendizaje - {model_name}')
#         plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f'learning_curves_{model_name.lower().replace(" ", "_")}.png'))
#         plt.close()


# def entrenar_comparar_modelos(X_train, y_train, X_test, y_test, label_encoder):
#     """
#     Entrena y compara los dos modelos: Keras (recalibrado) y Scikit-learn (tuneado).
#     """
#     clase_critica = 0
#     clase_no_critica = 1
    
#     # --- Modelo 1: TensorFlow/Keras ---
#     print("\n--- 4a. Modelado con TensorFlow/Keras (Recalibrado) ---")
    
#     # AJUSTE DE PESOS DE CLASE para un mejor equilibrio
#     class_weights_dict = {clase_critica: 1.7, clase_no_critica: 1}

#     model_keras = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(X_train.shape[1],)),
#         tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#     model_keras.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    
#     history = model_keras.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
#                               class_weight=class_weights_dict,
#                               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
#                               verbose=1)

#     # LÓGICA DE PREDICCIÓN CON UMBRAL AJUSTADO
#     prob_clase_no_critica = model_keras.predict(X_test)
#     prob_clase_critica = 1 - prob_clase_no_critica
#     y_pred_keras = np.where(prob_clase_critica >= KERAS_PREDICTION_THRESHOLD, clase_critica, clase_no_critica)
#     evaluar_modelo("TensorFlow-Keras", y_test, y_pred_keras, label_encoder, history)

#     # --- Modelo 2: Scikit-learn MLP (con tuneo avanzado) ---
#     print("\n--- 4b/5. Modelado y Tuneo Avanzado con Scikit-learn MLP ---")
#     pipeline_sklearn_smote = ImbPipeline([
#         ('smote', SMOTE(random_state=RANDOM_STATE)),
#         ('mlp', MLPClassifier(random_state=RANDOM_STATE, max_iter=300, early_stopping=True))
#     ])
#     param_grid = {
#         'mlp__hidden_layer_sizes': [(64, 32), (100,)],
#         'mlp__activation': ['relu', 'tanh'],
#         'mlp__alpha': [0.001, 0.005],
#         'mlp__learning_rate_init': [0.001, 0.0005]
#     }
#     print("Iniciando búsqueda con GridSearchCV (optimizando para F1-score)...")
#     search = GridSearchCV(pipeline_sklearn_smote, param_grid, n_jobs=-1, cv=3, scoring='f1_macro', verbose=1)
#     search.fit(X_train, y_train)
    
#     print("\nMejores parámetros encontrados por GridSearchCV (basado en F1-score):")
#     print(search.best_params_)
#     model_sklearn_tuned = search.best_estimator_
#     y_pred_sklearn = model_sklearn_tuned.predict(X_test)
#     evaluar_modelo("Scikit-learn MLP (Tuned)", y_test, y_pred_sklearn, label_encoder)
    
#     return model_keras, model_sklearn_tuned


# if __name__ == '__main__':
#     # Flujo de ejecución principal
#     df_final = cargar_y_preparar_datos(DATA_PATH)
#     analisis_exploratorio(df_final)
#     X_train_p, X_test_p, y_train, y_test, preprocessor, label_encoder = preprocesar_y_dividir(df_final)
    
#     print("\nClases detectadas por LabelEncoder:")
#     print(f"La clase '0' es '{label_encoder.classes_[0]}'")
#     print(f"La clase '1' es '{label_encoder.classes_[1]}'")
    
#     model_keras, model_sklearn = entrenar_comparar_modelos(X_train_p, y_train, X_test_p, y_test, label_encoder)

#     # Selección final y guardado de artefactos
#     print("\n--- Selección Final y Guardado de Artefactos ---")
#     print("Se seleccionó el modelo de Keras como el final para el despliegue por su equilibrio entre recall y precisión.")
    
#     model_keras.save(os.path.join(MODEL_OUTPUT_DIR, 'dengue_model.h5'))
#     joblib.dump(preprocessor, os.path.join(MODEL_OUTPUT_DIR, 'preprocessor.pkl'))
#     joblib.dump(label_encoder, os.path.join(MODEL_OUTPUT_DIR, 'label_encoder.pkl'))

#     print(f"\n¡PROCESO COMPLETADO! El modelo final y los artefactos se han guardado en la carpeta '{MODEL_OUTPUT_DIR}'.")
# ==============================================================================
# SCRIPT DE ENTRENAMIENTO FINAL - VERSIÓN ACTUALIZADA
#
# CAMBIOS:
# - Se filtran los registros de sexo 'I' para entrenar solo con 'M' y 'F'.
# - Se elimina la columna 'nombre_municipio_procedencia' del modelo.
# ==============================================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import random

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# --- CONFIGURACIÓN GLOBAL ---
DATA_PATH = 'Dengue.csv'
MODEL_OUTPUT_DIR = 'modelo_final_dengue'
TARGET = 'clasificacion_final'
RANDOM_STATE = 42
KERAS_PREDICTION_THRESHOLD = 0.45

# --- FIJAR SEMILLAS PARA REPRODUCIBILIDAD ---
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def cargar_y_preparar_datos(path):
    """
    Carga los datos, realiza ingeniería de características, filtra datos no deseados
    y selecciona las columnas finales del modelo.
    """
    print("--- 1. Cargando y Preparando Datos ---")
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin1')
    
    df.columns = df.columns.str.strip()
    
    # >> CAMBIO 1: Filtrar los datos para mantener solo sexo 'M' y 'F'
    print(f"Forma original del dataset: {df.shape}")
    df = df[df['sexo'].isin(['M', 'F'])]
    print(f"Forma después de filtrar sexo 'I': {df.shape}")

    # Ingeniería de Características
    for col in ['fecha_nototificacion', 'fecha_consulta', 'fecha_inicio_sintomas']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df = df.dropna(subset=['fecha_consulta', 'fecha_inicio_sintomas'])
    df['dias_hasta_consulta'] = (df['fecha_consulta'] - df['fecha_inicio_sintomas']).dt.days
    df = df[(df['dias_hasta_consulta'] >= 0) & (df['dias_hasta_consulta'] < 30)]
    df['mes_sintomas'] = df['fecha_inicio_sintomas'].dt.month

    df[TARGET] = df[TARGET].replace({
        'DENGUE GRAVE': 'Con signos de alarma', 
        'DENGUE CON SIGNOS DE ALARMA': 'Con signos de alarma',
        'DENGUE SIN SIGNOS DE ALARMA': 'Sin signos de alarma'
    })
    
    # >> CAMBIO 2: Eliminar la columna de provincia/departamento de la lista
    final_model_columns = [
        'edad', 'sexo', 'fiebre', 'cefalea', 
        'malgias', 'artralgia', 'erupcion', 'dias_hasta_consulta', 'mes_sintomas',
        TARGET
    ]
    
    df_final = df.reindex(columns=final_model_columns)
    df_final = df_final.dropna().reset_index(drop=True)

    print(f"\n[INFO] Forma del DataFrame final listo para entrenar: {df_final.shape}")
    if df_final.empty:
        print("[ERROR CRÍTICO] El DataFrame está vacío.")
        exit()

    for col in df_final.select_dtypes(include='object').columns:
        if col != TARGET:
            df_final[col] = df_final[col].astype(str)
            
    return df_final


def analisis_exploratorio(df):
    print("\n--- 2. Análisis Exploratorio de Datos (EDA) ---")
    print("Distribución de la variable objetivo:")
    print(df[TARGET].value_counts(normalize=True))
    plt.figure(figsize=(8, 5))
    sns.countplot(x=TARGET, data=df, palette='viridis')
    plt.title('Distribución de Clases (Dengue)')
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, 'distribucion_clases.png'))
    plt.close()

def preprocesar_y_dividir(df):
    print("\n--- 3. Preprocesamiento y División de Datos ---")
    X = df.drop(columns=[TARGET])
    y_raw = df[TARGET]
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_OUTPUT_DIR, 'model_columns.pkl'))
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, label_encoder

def evaluar_modelo(model_name, y_test, y_pred, label_encoder, history=None):
    print(f"\n--- 6. Evaluación del Modelo: {model_name} ---")
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(report)
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()
    if history:
        pd.DataFrame(history.history).plot(figsize=(10, 6))
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.title(f'Curvas de Aprendizaje - {model_name}')
        plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f'learning_curves_{model_name.lower().replace(" ", "_")}.png'))
        plt.close()

def entrenar_comparar_modelos(X_train, y_train, X_test, y_test, label_encoder):
    clase_critica = 0
    clase_no_critica = 1
    print("\n--- 4a. Modelado con TensorFlow/Keras (Recalibrado) ---")
    class_weights_dict = {clase_critica: 1.5, clase_no_critica: 1}
    model_keras = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_keras.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    history = model_keras.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
                              class_weight=class_weights_dict,
                              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                              verbose=1)
    prob_clase_no_critica = model_keras.predict(X_test)
    prob_clase_critica = 1 - prob_clase_no_critica
    y_pred_keras = np.where(prob_clase_critica >= KERAS_PREDICTION_THRESHOLD, clase_critica, clase_no_critica)
    evaluar_modelo("TensorFlow-Keras", y_test, y_pred_keras, label_encoder, history)
    print("\n--- 4b/5. Modelado y Tuneo Avanzado con Scikit-learn MLP ---")
    pipeline_sklearn_smote = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('mlp', MLPClassifier(random_state=RANDOM_STATE, max_iter=300, early_stopping=True))
    ])
    param_grid = {'mlp__hidden_layer_sizes': [(64, 32), (100,)], 'mlp__activation': ['relu', 'tanh'],
                  'mlp__alpha': [0.001, 0.005], 'mlp__learning_rate_init': [0.001, 0.0005]}
    print("Iniciando búsqueda con GridSearchCV (optimizando para F1-score)...")
    search = GridSearchCV(pipeline_sklearn_smote, param_grid, n_jobs=-1, cv=3, scoring='f1_macro', verbose=1)
    search.fit(X_train, y_train)
    print("\nMejores parámetros encontrados por GridSearchCV (basado en F1-score):")
    print(search.best_params_)
    model_sklearn_tuned = search.best_estimator_
    y_pred_sklearn = model_sklearn_tuned.predict(X_test)
    evaluar_modelo("Scikit-learn MLP (Tuned)", y_test, y_pred_sklearn, label_encoder)
    return model_keras, model_sklearn_tuned

if __name__ == '__main__':
    df_final = cargar_y_preparar_datos(DATA_PATH)
    analisis_exploratorio(df_final)
    X_train_p, X_test_p, y_train, y_test, preprocessor, label_encoder = preprocesar_y_dividir(df_final)
    print("\nClases detectadas por LabelEncoder:")
    print(f"La clase '0' es '{label_encoder.classes_[0]}'")
    print(f"La clase '1' es '{label_encoder.classes_[1]}'")
    model_keras, model_sklearn = entrenar_comparar_modelos(X_train_p, y_train, X_test_p, y_test, label_encoder)
    print("\n--- Selección Final y Guardado de Artefactos ---")
    print("Se seleccionó el modelo de Keras como el final para el despliegue.")
    model_keras.save(os.path.join(MODEL_OUTPUT_DIR, 'dengue_model.h5'))
    joblib.dump(preprocessor, os.path.join(MODEL_OUTPUT_DIR, 'preprocessor.pkl'))
    joblib.dump(label_encoder, os.path.join(MODEL_OUTPUT_DIR, 'label_encoder.pkl'))
    print(f"\n¡PROCESO COMPLETADO! El modelo final se ha guardado en '{MODEL_OUTPUT_DIR}'.")