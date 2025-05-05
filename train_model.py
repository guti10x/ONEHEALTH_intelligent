import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta, time
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import seaborn as sns

# Inicializar colorama
init(autoreset=True)

# Funciones para imprimir con colores
def info(msg): print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {msg}")
def success(msg): print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {msg}")
def warning(msg): print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {msg}")
def error(msg): print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")

# --------------------------
# 1. Conexión a Firebase
# --------------------------
print('\n' + '=' * 60)
info('Paso 1: Conectando a Firebase...')
cred = credentials.Certificate('./credentials_firebase/onehealth-f4967-firebase-adminsdk-fbsvc-e899f7b095.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
success('Conexión a Firebase realizada exitosamente ✔')

# ------------------------------------------
# 2. Obtención de formularios de Firestore
# ------------------------------------------
print('\n' + '=' * 60)
info('Paso 2: Obteniendo datos de la colección "formularios"...')
formularios = []
try:
    doc_ref = db.collection('formularios')
    docs = doc_ref.stream()

    for doc in docs:
        form_data = doc.to_dict()
        form_data['doc_id'] = doc.id
        formularios.append(form_data)

    success(f'{len(formularios)} formularios encontrados y cargados correctamente ✔')
    print(f'Formularios: {formularios}')
except Exception as e:
    error(f'Fallo al leer los formularios desde Firestore: {e}')
    formularios = []

# ------------------------------------------------
# 3. Obtención de datos biométricos de Firestore
# ------------------------------------------------
print('\n' + '=' * 60)
info('Paso 3: Obteniendo datos de biométricos de "biometric_data"...')
error('Not implemented yet...')

# ----------------------------------------------------------------------------------------
# 4. Preprocesamiento de datos de formularios (catalogación en formulario de mañana o noche)
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 4: Procesando formularios...')

if formularios:
    df = pd.DataFrame(formularios)

    # Convertir recorded_at a datetime y ajustar zona horaria
    print(Fore.CYAN + '[INFO] Ajustando zona horaria de recorded_at...')
    try:
        df['recorded_at'] = pd.to_datetime(df['recorded_at']).dt.tz_convert('Europe/Madrid')
        success('Fechas convertidas y ajustadas a zona horaria Europe/Madrid ✔')
    except Exception as e:
        error(f'Error al convertir las fechas: {e}')

    # Función para clasificar el período
    print(Fore.CYAN + '[INFO] Clasificando formularios en "mañana" y "noche"...')
    def classify_period(row):
        hour = row['recorded_at'].hour
        minute = row['recorded_at'].minute
        if time(6, 0) <= time(hour, minute) < time(19, 0):
            return 'mañana'
        else:
            return 'noche'

    # Aplicar clasificación
    df['period'] = df.apply(classify_period, axis=1)

    # Separar en dos DataFrames
    df_morning = df[df['period'] == 'mañana'].copy()
    df_night = df[df['period'] == 'noche'].copy()

    if not df_morning.empty and not df_night.empty:
        success('Formularios clasificados y DataFrames creados correctamente ✔')
    else:
        warning('Algunos de los DataFrames "mañana" o "noche" están vacíos.')


else:
    warning('No se encontraron formularios para procesar.')

# ----------------------------------------------------------------------------------------
# 5. Procesamiento de datos:
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 5: Procesando datos...')

for df_label, df_ref in [('mañana', df_morning), ('noche', df_night)]:
    print(Fore.CYAN + f'\n[INFO] Procesando formularios de: {df_label.upper()}')

    # -----------------------------------------------------------------------------
    # 5.1 Eliminar Columnas No Relevantes
    # -----------------------------------------------------------------------------
    columns_to_drop = ['id_user', 'recorded_at', 'period']
    df_ref.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # -----------------------------------------------------------------------------
    # 5.2 Eliminar Filas con Valores Nulos
    # -----------------------------------------------------------------------------
    critical_columns = ['sadnessLevel', 'happinessLevel', 'avgEnergyLevel']
    df_ref.dropna(subset=critical_columns, inplace=True)

    # -----------------------------------------------------------------------------
    # 5.3 Convertir Variables Categóricas a Numéricas
    # -----------------------------------------------------------------------------
    df_ref = pd.get_dummies(df_ref, columns=['country', 'state', 'city'], drop_first=True)

    # -----------------------------------------------------------------------------
    # 5.4 Detectar y Eliminar Duplicados
    # -----------------------------------------------------------------------------
    df_ref.drop_duplicates(inplace=True)

    # -----------------------------------------------------------------------------
    # 5.5 Análisis y Manejo de Outliers
    # -----------------------------------------------------------------------------
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    numeric_cols = ['sadnessLevel', 'avgEnergyLevel', 'maxAnxietyLevel', 'happinessLevel']
    for col in numeric_cols:
        if col in df_ref.columns:
            df_ref = remove_outliers(df_ref, col)

    # -----------------------------------------------------------------------------
    # 5.5 Crear Nuevas Variables Calculadas
    # -----------------------------------------------------------------------------

    # ----------------------------------------------
    # 5.5.1 Calcular Tiempo Total de Redes Sociales 
    # ----------------------------------------------
    if 'instagram_time' in df_ref.columns and 'tiktok_time' in df_ref.columns:
        df_ref['total_social_media_time'] = df_ref['instagram_time'] + df_ref['tiktok_time']
        success(f'[{df_label.upper()}] Variable del tiempo total invertido en redes sociales creada correctamente ✔')
    else:
        error(f'[{df_label.upper()}] Columnas necesarias para calcular el tiempo total de redes sociales no encontradas.')

    # ----------------------------------------------
    # 5.5.2 Extraer app más usada, segunda más usada y tercera más usada
    # ----------------------------------------------
    if 'final_ranking' in df_ref.columns:

        def extract_top_apps(ranking_string, position):
            try:
                apps = ranking_string.split(',')
                return apps[position] if len(apps) > position else None
            except Exception:
                return None

        df_ref['top1_app'] = df_ref['final_ranking'].apply(lambda x: extract_top_apps(x, 0))
        df_ref['top2_app'] = df_ref['final_ranking'].apply(lambda x: extract_top_apps(x, 1))
        df_ref['top3_app'] = df_ref['final_ranking'].apply(lambda x: extract_top_apps(x, 2))
        success(f'[{df_label.upper()}] Aplicación más usada, segunda más usada y tercera más usada creadas correctamente ✔')
    else:
        error(f'[{df_label.upper()}] Columna "final_ranking" no encontrada para extraer Top apps.')

    # ----------------------------------------------
    # 5.5.3 Calcular Promedio de Estado de Ánimo
    # ----------------------------------------------
    mood_cols = ['happinessLevel', 'sadnessLevel', 'apathyLevel', 'avgAnxietyLevel', 'avgEnergyLevel']
    existing_mood_cols = [col for col in mood_cols if col in df_ref.columns]
    if existing_mood_cols:
        df_ref['average_mood'] = df_ref[existing_mood_cols].mean(axis=1)
        success(f'[{df_label.upper()}] Variable estado de ánimo promedio creada correctamente ✔')
    else:
        error(f'[{df_label.upper()}] No se encontraron columnas necesarias para calcular el estado de ánimo promedio.')

    # ----------------------------------------------
    # 5.5.4 Calcular cantidad de horas de sueño
    # ----------------------------------------------
    if 'sleep_time' in df_ref.columns and 'wake_up_time' in df_ref.columns:
        try:
            df_ref['sleep_time'] = pd.to_datetime(df_ref['sleep_time'])
            df_ref['wake_up_time'] = pd.to_datetime(df_ref['wake_up_time'])
            df_ref['sleep_duration_hours'] = (df_ref['wake_up_time'] - df_ref['sleep_time']).dt.total_seconds() / 3600
            success(f'[{df_label.upper()}] Variable duración del sueño creada correctamente ✔')
        except Exception as e:
            error(f'[{df_label.upper()}] Error calculando duración del sueño: {e}')
    else:
        error(f'[{df_label.upper()}] Columnas necesarias para calcular la duración del sueño no encontradas.')

# ----------------------------------------------------------------------------------------
# 6. Guardar DataFrames procedo resultantes
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 6: Guardando DataFrames en archivos CSV...')

try:
    df_morning.to_csv('./output/df_morning.csv', index=False, encoding='utf-8-sig')
    success('DataFrame de mañana guardado exitosamente en ./output/df_morning.csv ✔')
except Exception as e:
    error(f'Error al guardar DataFrame "mañana": {e}')

try:
    df_night.to_csv('./output/df_night.csv', index=False, encoding='utf-8-sig')
    success('DataFrame de noche guardado exitosamente en ./output/df_night.csv ✔')
except Exception as e:
    error(f'Error al guardar DataFrame "noche": {e}')

# ----------------------------------------------------------------------------------------
# 7. Analisis Exploratorio de datos y visualización
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# 7. Análisis Exploratorio de Datos y Visualización
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 7: Realizando análisis exploratorio de datos y visualización...')

# ----------------------------------------------------------------------------------------
# 7.1 Matriz de Correlación
# ----------------------------------------------------------------------------------------
print(Fore.CYAN + '[INFO] Identificando columnas numéricas en los DataFrames...')
numeric_columns_morning = df_morning.select_dtypes(include=['number']).columns
numeric_columns_night = df_night.select_dtypes(include=['number']).columns

# df_morning
print(Fore.CYAN + '[INFO] Calculando matriz de correlación para df_morning...')
morning_corr = df_morning[numeric_columns_morning].corr()
print(Fore.CYAN + '[INFO] Visualizando matriz de correlación para df_morning...')
plt.figure(figsize=(10, 8))
sns.heatmap(morning_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matriz de Correlación - Mañana")
plt.show()

# df_night
print(Fore.CYAN + '[INFO] Calculando matriz de correlación para df_night...')
night_corr = df_night[numeric_columns_night].corr()
print(Fore.CYAN + '[INFO] Visualizando matriz de correlación para df_night...')
plt.figure(figsize=(10, 8))
sns.heatmap(night_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matriz de Correlación - Noche")
plt.show()

# ----------------------------------------------------------------------------------------
# 7.2 Histogramas
# ----------------------------------------------------------------------------------------
def plot_histograms(df, title):
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].dropna().shape[0] > 0:
            print(Fore.CYAN + f'[INFO] Generando histograma para {col} - {title}...')
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True, bins=20)
            plt.title(f'Histograma de {col} - {title}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.show()
        else:
            print(Fore.YELLOW + f'[ADVERTENCIA] No se puede generar histograma para {col} en {title} debido a datos insuficientes.')

# Aplicación de histogramas
plot_histograms(df_morning, "Mañana")
plot_histograms(df_night, "Noche")

# ----------------------------------------------------------------------------------------
# 7.3 Boxplots
# ----------------------------------------------------------------------------------------
def plot_boxplots(df, title):
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].dropna().shape[0] > 1:
            print(Fore.CYAN + f'[INFO] Generando boxplot para {col} - {title}...')
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot de {col} - {title}')
            plt.xlabel(col)
            plt.show()
        else:
            print(Fore.YELLOW + f'[ADVERTENCIA] No se puede generar boxplot para {col} en {title} debido a datos insuficientes.')

# Aplicación de boxplots
plot_boxplots(df_morning, "Mañana")
plot_boxplots(df_night, "Noche")

# ----------------------------------------------------------------------------------------
# 7.4 Scatter Plots
# ----------------------------------------------------------------------------------------
def plot_scatter(df, x_col, y_col, title):
    if df[x_col].dropna().shape[0] > 0 and df[y_col].dropna().shape[0] > 0:
        print(Fore.CYAN + f'[INFO] Generando scatter plot entre {x_col} y {y_col} - {title}...')
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f'Relación entre {x_col} y {y_col} - {title}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()
    else:
        print(Fore.YELLOW + f'[ADVERTENCIA] No se puede generar scatter plot para {x_col} y {y_col} en {title} debido a datos insuficientes.')

# Scatter plots de ejemplo (puedes cambiar las columnas)
if len(numeric_columns_morning) >= 2:
    plot_scatter(df_morning, numeric_columns_morning[0], numeric_columns_morning[1], "Mañana")

if len(numeric_columns_night) >= 2:
    plot_scatter(df_night, numeric_columns_night[0], numeric_columns_night[1], "Noche")

# ----------------------------------------------------------------------------------------
# 7.5 Pairplot 
# ----------------------------------------------------------------------------------------
def plot_pairplot(df, title):
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        print(Fore.CYAN + f'[INFO] Generando pairplot para múltiples variables numéricas - {title}...')
        sns.pairplot(df[numeric_cols])
        plt.suptitle(f'Pairplot - {title}', y=1.02)
        plt.show()
    else:
        print(Fore.YELLOW + f'[ADVERTENCIA] No se puede generar pairplot para {title} debido a columnas insuficientes.')

plot_pairplot(df_morning, "Mañana")
plot_pairplot(df_night, "Noche")


# ----------------------------------------------------------------------------------------
# Entrenar modelo
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Evaluar modelo
# ----------------------------------------------------------------------------------------
