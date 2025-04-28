import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta

# --------------------------
# 1. Conexión a Firebase
# --------------------------
print('-' * 60)
print('CONECTANDO A FIREBASE...')
cred = credentials.Certificate('./credentials_firebase/onehealth-f4967-firebase-adminsdk-fbsvc-e899f7b095.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
print('Conexión a Firebase realizada exitosamente ✔')

# ------------------------------------------
# 2. Obtención de formularios de Firestore
# ------------------------------------------
print('-' * 60)
print('OBTENIENDO DATOS DE FIREBASE...')
formularios = []
try:
    doc_ref = db.collection('formularios')
    docs = doc_ref.stream()

    for doc in docs:
        form_data = doc.to_dict()
        form_data['doc_id'] = doc.id
        formularios.append(form_data)

    print(f'{len(formularios)} formularios encontrados y cargados correctamente ✔')
except Exception as e:
    print('Error leyendo Firestore:', e)
    formularios = []

# ------------------------------------------
# 2. Obtención de datos biométricos de Firestore
# ------------------------------------------
# Pending

# -------------------------------------
# 3. Preprocesar datos de formularios
# -------------------------------------
if formularios:
    # Creamos dataframe general
    df = pd.DataFrame(formularios)

    # Comporbamos que recorded_at tiene formato datetime y zona horaria correcta
    df['recorded_at'] = pd.to_datetime(df['recorded_at'])
    madrid_tz = pytz.timezone('Europe/Madrid')
    df['recorded_at'] = df['recorded_at'].dt.tz_convert(madrid_tz)

    # Clasificar el formulario entre mañana(6am-19pm)/noche(19pm-6am)
    def classify_period(recorded_at):
        hour = recorded_at.hour
        if 6 <= hour < 19:
            return 'mañana'
        else:
            return 'noche'

    df['period'] = df['recorded_at'].apply(classify_period)

    # Crear día lógico
    def logical_day(recorded_at, period):
        if period == 'noche' and recorded_at.hour < 6:
            return (recorded_at - timedelta(days=1)).date()
        else:
            return recorded_at.date()

    df['logical_day'] = df.apply(lambda row: logical_day(row['recorded_at'], row['period']), axis=1)

    # --------------------------------------------
    # 4. Ordenar y unir formularios mañana-noche
    # --------------------------------------------
    # Formularios de mañana y noche consecutivos matcheados por id_user y logical_day
    merged_rows = []

    # Agrupamos los DataFrame por id_user y luego por logical_day
    # Para cada usuario y día lógico, seleccionamos el primer registro de la mañana y el primer registro de la noche ordenados por recorded_at
    # Combinamos estos dos registros en un solo y lo agrega a la lista merged_rows
    for user_id, group in df.groupby('id_user'):
        for day, day_group in group.groupby('logical_day'):
            morning = day_group[day_group['period'] == 'mañana']
            night = day_group[day_group['period'] == 'noche']

            if len(morning) > 0 and len(night) > 0:
                morning_row = morning.sort_values('recorded_at').iloc[0]
                night_row = night.sort_values('recorded_at').iloc[0]

                merged = {f'morning_{col}': morning_row[col] for col in morning_row.index}
                merged.update({f'night_{col}': night_row[col] for col in night_row.index})
                merged_rows.append(merged)

    merged_df = pd.DataFrame(merged_rows)
    print(f'{len(merged_df)} pares de formularios mañana-noche combinados ✔')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(merged_df)

    # -----------------------------------------
    # 5. Procesamiento del dataframe fusuinado
    # -----------------------------------------
    print('-' * 60)
    print('PREPROCESANDO DATOS DEL DATAFRAME...')

    # Tomar solo los datos de interés (excluyendo metadata como 'doc_id', etc.)
    formularios_data = [f for f in formularios]
    processed_df = pd.DataFrame(formularios_data)

    print(processed_df)

    # ----------------------------------------
    # 5.1 Eliminamos columnas no relevantes
    # ----------------------------------------
    print('Eliminando columnas NO relevantes...')
    columns_to_drop = ['id_user', 'doc_id']
    columns_to_remove = [col for col in columns_to_drop if col in processed_df.columns]
    if columns_to_remove:
        print(f'Columnas eliminadas: {columns_to_remove} ✔\n')
        processed_df = processed_df.drop(columns=columns_to_remove)

    # ----------------------------------------   
    # 5.2 Eliminamos filas con valores nulos
    # ----------------------------------------
    print('Eliminando filas con valores nulos...')
    nulos = processed_df[processed_df.isnull().any(axis=1)]
    if not nulos.empty:
        print(f'Se eliminaron {len(nulos)} filas con nulos ✔\n')
        processed_df = processed_df.dropna()
    else:
        print('No se encontraron filas nulas ✔\n')

    # ---------------------------------------------------
    # 5.3 Convertimos variables categóricas a numéricas
    # ---------------------------------------------------
    print('Convirtiendo variables categóricas a numéricas...')
    processed_df = pd.get_dummies(processed_df)
    print('Conversión completada ✔\n')

    # ----------------------------
    # 5.4 Detección de duplicados
    # ----------------------------
    print('Detectando duplicados...')
    duplicados = processed_df.duplicated().sum()
    if duplicados > 0:
        print(f'{duplicados} duplicados encontrados, eliminando...')
        processed_df = processed_df.drop_duplicates()
    else:
        print('No duplicados encontrados ✔\n')

    # --------------------------
    # 5.5 Análisis de outliers
    # --------------------------
    print('Analizando outliers...')
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    Q1 = processed_df[numeric_cols].quantile(0.25)
    Q3 = processed_df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((processed_df[numeric_cols] < (Q1 - 1.5 * IQR)) | (processed_df[numeric_cols] > (Q3 + 1.5 * IQR)))
    num_outliers = outliers.sum().sum()
    if num_outliers > 0:
        print(f'Detectados {num_outliers} valores atípicos.')
    else:
        print('No se detectaron outliers ✔\n')

    # ----------------------------------------------------------------------------------------
    # 5.6 Creación de nuevas variables calculadas de los datos que tenemos en los formularios
    # ----------------------------------------------------------------------------------------
    print('Creando nuevas variables...')

    # -------------------------------
    # Tiempo total en redes sociales
    # -------------------------------
    if 'instagram_time' in processed_df.columns and 'tiktok_time' in processed_df.columns:
        processed_df['total_social_media_time'] = processed_df['instagram_time'] + processed_df['tiktok_time']
        print('Variable del tiempo total invertido en redes sociales creada correctamente ✔')

    # ---------------------------------------
    # Extracción de top 1, top 2, top 3 apps
    # ---------------------------------------
    if 'final_ranking' in processed_df.columns:
        print('Extrayendo Top apps...')

        def extract_top_apps(ranking_string, position):
            try:
                apps = ranking_string.split(',')
                return apps[position] if len(apps) > position else None
            except Exception:
                return None

        processed_df['top1_app'] = processed_df['final_ranking'].apply(lambda x: extract_top_apps(x, 0))
        processed_df['top2_app'] = processed_df['final_ranking'].apply(lambda x: extract_top_apps(x, 1))
        processed_df['top3_app'] = processed_df['final_ranking'].apply(lambda x: extract_top_apps(x, 2))
        print('Aplicaión mas usada, segunda más usada y tercera más usada creada correctamente ✔')

    # --------------------------
    # Estado de ánimo promedio
    # --------------------------
    mood_cols = ['happinessLevel', 'sadnessLevel', 'apathyLevel', 'avgAnxietyLevel', 'avgEnergyLevel']
    existing_mood_cols = [col for col in mood_cols if col in processed_df.columns]
    if existing_mood_cols:
        processed_df['average_mood'] = processed_df[existing_mood_cols].mean(axis=1)
        print('Variable estado de ánimo promedio creada correctamente ✔')

    # --------------------------
    # Duración del sueño
    # --------------------------
    if 'sleep_time' in processed_df.columns and 'wake_up_time' in processed_df.columns:
        try:
            processed_df['sleep_time'] = pd.to_datetime(processed_df['sleep_time'])
            processed_df['wake_up_time'] = pd.to_datetime(processed_df['wake_up_time'])
            processed_df['sleep_duration_hours'] = (processed_df['wake_up_time'] - processed_df['sleep_time']).dt.total_seconds() / 3600
            print('Variable duracción del sueño creada correctamente ✔')
        except Exception as e:
            print(f'Error calculando duración de sueño: {e}')

else:
    print('No se encontraron formularios para procesar.')
