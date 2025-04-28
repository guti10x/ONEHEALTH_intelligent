import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
#from sklearn.preprocessing import StandardScaler
import numpy as np

### 1- Conexión con la base de datos de Firebase
print('-' * 60)
print('CONECTANDO A FIREBASE...')
cred = credentials.Certificate('./credentials_firebase/onehealth-f4967-firebase-adminsdk-fbsvc-e899f7b095.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
print('Conexión a Firebase realizada exitosamente ✔')

### 2- Obtención de todos los datos recopilados
print('-' * 60)
print('OBTENIENDO DATOS DE FIREBASE...')
formularios = []
try:
    doc_ref = db.collection('formularios')
    docs = doc_ref.stream()

    for doc in docs:
        formularios.append({'id': doc.id, 'data': doc.to_dict()})

    print(f'{len(formularios)} formularios encontrados y cargados correctamente ✔')
    print("Data received from Firestore:", formularios)
except Exception as e:
    print('Error leyendo Firestore:', e)

# De datos biométricos
biometric_data = [] 
# Pending

### 3- PREPROCESADO DE DATOS
print('-' * 60)
print('PREPROCESANDO DATOS...')

formularios_data = [f['data'] for f in formularios]
df = pd.DataFrame(formularios_data)
print(df)

# 3.1 Eliminamos columnas No relevantes
print('Eliminando columnas NO relevantes...')
columns_to_drop = ['id_user']
columns_to_remove = [col for col in columns_to_drop if col in df.columns]
if columns_to_remove:
    print(f'Columnas no relevantes eliminadas: {columns_to_remove} ✔\n')
    df = df.drop(columns=columns_to_remove)

# 3.2 Eliminamos filas con valores nulos
print('Eliminando filas con valores nulos...')
nulos = df[df.isnull().any(axis=1)]
if not nulos.empty:
    print(f'Se han eliminado {len(nulos)} filas con valores nulos ✔\n')
    df = df.dropna()
else:
    print('No se encontraron filas con valores nulos ✔\n')

# 3.3 Convertimos variables categóricas a variables numéricas
print('Convirtiendo variables categóricas a numéricas...')
df = pd.get_dummies(df)
print('Variables categóricas convertidas ✔\n')

# 3.4 Detección de duplicados
print('Detectando registros duplicados...')
duplicados = df.duplicated().sum()
if duplicados > 0:
    print(f'Se encontraron {duplicados} duplicados, eliminando...')
    df = df.drop_duplicates()
else:
    print('No se encontraron duplicados ✔\n')

# 3.5 Análisis de outliers
print('Analizando outliers...')
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR)))
num_outliers = outliers.sum().sum()

if num_outliers > 0:
    print(f'Detectados {num_outliers} valores atípicos')
    # Aquí solo reportamos, no los eliminamos por ahora
else:
    print('No se detectaron valores atípicos ✔\n')

# 3.6 Creación de nuevas variables
print('Creando nuevas variables calculadas...')

# 3.6.1 Tiempo total en redes sociales
if 'instagram_time' in df.columns and 'tiktok_time' in df.columns:
    df['total_social_media_time'] = df['instagram_time'] + df['tiktok_time']
    print('Variable "total_social_media_time" creada ✔')

# 3.6.2 Extracción de Top 1, Top 2 y Top 3 de apps
if 'final_ranking' in df.columns:
    print('Extrayendo Top 1, Top 2 y Top 3 apps...')
    
    def extract_top_apps(ranking_string, position):
        try:
            apps = ranking_string.split(',')
            return apps[position] if len(apps) > position else None
        except Exception:
            return None

    df['top1_app'] = df['final_ranking'].apply(lambda x: extract_top_apps(x, 0))
    df['top2_app'] = df['final_ranking'].apply(lambda x: extract_top_apps(x, 1))
    df['top3_app'] = df['final_ranking'].apply(lambda x: extract_top_apps(x, 2))

    print('Variables "top1_app", "top2_app" y "top3_app" creadas ✔')

# 3.6.3 Estado de ánimo promedio
mood_cols = ['happinessLevel', 'sadnessLevel', 'apathyLevel', 'avgAnxietyLevel', 'avgEnergyLevel']
existing_mood_cols = [col for col in mood_cols if col in df.columns]
if existing_mood_cols:
    df['average_mood'] = df[existing_mood_cols].mean(axis=1)
    print('Variable calculada para el nivel medio de ánimo creada correctamente ✔')

# 3.6.4 Duración del sueño
if 'sleep_time' in df.columns and 'wake_up_time' in df.columns:
    try:
        df['sleep_time'] = pd.to_datetime(df['sleep_time'])
        df['wake_up_time'] = pd.to_datetime(df['wake_up_time'])
        df['sleep_duration_hours'] = (df['wake_up_time'] - df['sleep_time']).dt.total_seconds() / 3600
        print('Variable calculada para clacular el numero de horas dormidas creada ✔')
    except Exception as e:
        print(f'Error calculando duración de sueño: {e}')
