import firebase_admin
from firebase_admin import credentials, firestore
from colorama import Fore, Style, init
import joblib
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle

# Inicializar colorama
init(autoreset=True)

# Funciones para imprimir con colores
def info(msg): print(f"{Fore.BLUE}[INFO]{Style.RESET_ALL} {msg}")
def success(msg): print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {msg}")
def warning(msg): print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {msg}")
def error(msg): print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")

# -----------------------------------------
# 0. Mostrar Header del Script en terminal
# -----------------------------------------
print("-" * 100)
print(Fore.BLUE + """
             ..:^^^^^:.   .                        .:^^^^:.    .::::.     .::::.    ::::::::::::.  
         .~7JY55PPPPP55J7~^~~^.                 :!J55PPPPP5J!: ?P55557.   !PPPY:   :5PPPPPPPPPPP!  
      .~?5PPPP55555555PPP5J^~77!^.            .?5PPPPP55PPPPP57JPPPPPP5!  ^7!^.    ^PPPP55555555!  
     ~YPP55555555555555555PY:~7777~:         .JPPPP5!:..^?PPPPPPPPPPPPPPY^         ^PPPP7^^^^^^^:  
   .?PP55555555555555555555P!:777777~.       ^PPPPP!      JPPPPPPPPPPPPPPPJ??J^    ^PPPPPPPPPPPP!  
  :YP55555555555555555555555^^7777777!.      ^PPPPP?     :YPPPPPPPPPY!5PPPPPPP~    ^PPPPJ???????^  
 .JP5555PPPPPPPP55555555PP5~:777777777!.      7PPPPPY7!!?5PPPP55PPPPJ :?PPPPPP~    ~PPPP?!!!!!!!:  
 !P55P5YJ?777?JY55PPPPP5Y7^^77777777777^       ~JPPPPPPPPPPP57:YPPPPJ   ^JPPPP^    ~PPPPPPPPPPPG~  
 JPP57~~!7???7!~^^~~~~^::~!777777777777!         :!?JJYYJ?7^.  7????!     ~???:    ^????????????:  
.YPJ^!JYYYYYYYYYJ7.   .!777777777777777!                                                           
 JY:?YYYYYYYYYYYYYJ: :77777777777777777~     ...   ... .......    ...    ...   .:....:. ...  ...   
 ~~~YJYYYYYYYYYYYJY7 !77777777777777777.     ^!!. .!!^ !!!~~~~   ^!!!^   ~!~   :~~!!!~~.~!~. ^!!.  
   !YJYYYYYYYYYYYJYJ.~7777777777777777:      ^!!~~~!!^ !!~^~~^  ^!!^!!:  ~!~     .!!^   ~!!~~!!!.  
   :JYYYYYYYYYYYYYJY7:!7777777777777~.       ^!!:.:!!:.!!~^^^^ :!!!~!!!: ~!!^^^: .!!^   ~!~..^!!.  
    ^JYYJYYYYYYYYJYYY?^^!77777777!~:         :^^. .^^. ^^^^^^^.^^:...^^^.:^^^^^: .^^:   ^^:  :^^.  
     .7JYYYYYYYYYYYYJYY?!~~~~~~~!^                                                                 
       :!JYYYYYYYYYYJJYYYYJJJJ?!^                     OneHealth Intelligent V1 - Predict                                           
         .^!7?JJYYYYYYYYJJ?7~:.                                                                    
             ..:^^~~~~^::.                                                                                                                                                                                                                                                              
""")
print("-" * 100)
print(Fore.BLUE + '\n[INFO] Iniciando el script de predicción...')

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

# -----------------------------------------------------------------------------------------------------
# 2. Obtención de los datos para predecir (formularios rellenados en la pasada ventana (mañana/noche))
# -----------------------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 2: Obteniendo los últimos formularios rellenados del última ventana de formularios (mañana/noche)...')
prediction_period = None
try:
    # Obtener la fecha y hora actual
    now = datetime.now()
    today = now.date()

    # Determinar si vamos a predecir para noche o para mañana
    if 6 <= now.hour < 19:
        prediction_period = 'night'  # Estamos en mañana, predecimos para noche
        print(Fore.CYAN + f'[INFO] Obteniendo formularios de la mañana...')
    else:
        prediction_period = 'morning'  # Estamos en noche, predecimos para mañana
        print(Fore.CYAN + f'[INFO] Obteniendo formularios de la noche...')

    # Determinar el rango de tiempo en función del periodo de predicción
    if prediction_period == 'night':
        start_time = datetime.combine(today - timedelta(days=1), datetime.min.time()) + timedelta(hours=19)
        end_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=6)
    else:
        start_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=6)
        end_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=19)

    period = prediction_period  # Conservamos esta variable para compatibilidad

    print(Fore.YELLOW + f'Obteniendo formularios entre: {start_time} y {end_time} ({period})')

    # Filtrar formularios en el rango de tiempo
    docs = db.collection('formularios').where('recorded_at', '>=', start_time).where('recorded_at', '<', end_time).stream()
    data = [doc.to_dict() for doc in docs]
    df = pd.DataFrame(data)

    # Verificar si no se encontraron formularios
    if df.empty:
        raise ValueError(f"No se encontraron formularios entre {start_time} y {end_time} ({period})")

    # Guardar el id_user separado para mantener relación post-procesamiento
    id_users = df['id_user'].copy() if 'id_user' in df.columns else None

    # Imprimir los datos obtenidos de Firebase
    print(Fore.CYAN + '[DEBUG] Datos obtenidos de Firebase:')
    for i, doc in enumerate(data):
        print(f"Documento {i + 1}: {doc}")
    success(f'{len(df)} formularios encontrados y cargados correctamente ✔')
    print(Fore.YELLOW + f'[INFO] Datos cargados: {df.shape[0]} formulario(s), {df.shape[1]} columnas')
    print(Fore.YELLOW + f'[INFO] Columnas en datos crudos: {df.columns.tolist()}')
except Exception as e:
    error(f'Fallo al leer los formularios desde Firestore: {e}')
    exit()

# --------------------------------------
# 3. Carga de Modelos y Escaladores
# --------------------------------------
print('\n' + '=' * 60)
info('Paso 3: Cargando modelos, columnas y escaladores...')
models = {}
scalers = {}
training_columns = {}

# Definir rutas de archivos para cada periodo
night_files = {
    'model': 'predict_models_output/model_noche_Linear_Regression.pkl',
    'columns': 'predict_models_output/training_columns_noche.pkl',
    'scaler': 'predict_models_output/scaler_noche.pkl'
}

morning_files = {
    'model': 'predict_models_output/model_mañana_Linear_Regression.pkl',
    'columns': 'predict_models_output/training_columns_mañana.pkl',
    'scaler': 'predict_models_output/scaler_mañana.pkl'
}

# Cargar archivos según el periodo de predicción
if prediction_period == 'night':
    if os.path.exists(night_files['model']):
        models['night'] = joblib.load(night_files['model'])
        success('Modelo de noche cargado exitosamente ✔')
    else:
        warning('No se encontró el archivo del modelo de noche.')

    if os.path.exists(night_files['columns']):
        training_columns['night'] = joblib.load(night_files['columns'])
        success('Columnas de entrenamiento de noche cargadas exitosamente ✔')
    else:
        warning('No se encontraron las columnas de entrenamiento de noche.')

    if os.path.exists(night_files['scaler']):
        scalers['night'] = joblib.load(night_files['scaler'])
        success('Escalador de noche cargado exitosamente ✔')
    else:
        warning('No se encontró el escalador de noche.')

elif prediction_period == 'morning':
    if os.path.exists(morning_files['model']):
        models['morning'] = joblib.load(morning_files['model'])
        success('Modelo de mañana cargado exitosamente ✔')
    else:
        warning('No se encontró el archivo del modelo de mañana.')

    if os.path.exists(morning_files['columns']):
        training_columns['morning'] = joblib.load(morning_files['columns'])
        success('Columnas de entrenamiento de mañana cargadas exitosamente ✔')
    else:
        warning('No se encontraron las columnas de entrenamiento de mañana.')

    if os.path.exists(morning_files['scaler']):
        scalers['morning'] = joblib.load(morning_files['scaler'])
        success('Escalador de mañana cargado exitosamente ✔')
    else:
        warning('No se encontró el escalador de mañana.')

# --------------------------
# 4. Preprocesamiento de Datos
# --------------------------
print('\n' + '=' * 60)
info('Paso 4: Preprocesando datos...')

# Aquí ya no separo entre df_night y df_morning, trabajo con el df completo
print(f"[DEBUG] prediction_period: '{prediction_period}'")
print(f"[DEBUG] training_columns keys: {list(training_columns.keys())}")
print(f"[DEBUG] df rows: {df.shape[0]}")

# Guardar copia de los ID de usuario antes de eliminar columnas
id_users = df['id_user'].copy() if 'id_user' in df.columns else pd.Series([None] * len(df))

# Definición de la función de preprocesamiento de datos
def preprocess_data(df, training_cols, period):
    print('-' * 60)
    print(Fore.CYAN + f'\n[INFO] Preprocesando datos para {period} con {df.shape[0]} filas iniciales')

    # 1. Eliminar columnas no relevantes
    print('\n' + '.' * 60)
    print(Fore.CYAN + f'[INFO] Eliminando columnas no relevantes para {period}...')
    critical_columns = ['sadnessLevel', 'avgEnergyLevel', 'happinessLevel']
    try:
        columns_to_drop = ['id_user', 'doc_id', 'recorded_at', 'period', 'maxAnxietyLevel']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        success(f'Columnas no relevantes eliminadas para {period} ✔')
    except Exception as e:
        error(f'Error al eliminar columnas no relevantes: {e}')

    # 2. Rellenar columnas críticas con la media
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Rellenando valores faltantes en columnas críticas para {period}...')
    try:
        for col in critical_columns:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].mean())
        success(f'Valores faltantes en columnas críticas rellenados para {period} ✔')
    except Exception as e:
        error(f'Error al rellenar columnas críticas: {e}')

    # 3. One-hot encoding de variables categóricas
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Convirtiendo variables categóricas a numéricas mediante one-hot encoding...')
    try:
        categorical_cols = ['country', 'state', 'city', 'final_ranking']
        df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], dummy_na=False)
        success(f'Variables categóricas convertidas a numéricas para {period} ✔')
    except Exception as e:
        error(f'Error al convertir variables categóricas: {e}')

    # 4. Crear variable social_media_time
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Creando variable social_media_time...')
    try:
        if 'instagram_time' in df.columns and 'tiktok_time' in df.columns:
            df['social_media_time'] = df['instagram_time'] + df['tiktok_time']
            df['social_media_time'] = df['social_media_time'].fillna(df['social_media_time'].mean())
            success(f'Variable social_media_time creada para {period} ✔')
    except Exception as e:
        error(f'Error al crear social_media_time: {e}')

    # 5. Crear variable avg_emotion
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Creando variable avg_emotion...')
    try:
        if 'sadnessLevel' in df.columns and 'happinessLevel' in df.columns:
            df['avg_emotion'] = (df['sadnessLevel'] + df['happinessLevel']) / 2
            df['avg_emotion'] = df['avg_emotion'].fillna(df['avg_emotion'].mean())
            success(f'Variable avg_emotion creada para {period} ✔')
    except Exception as e:
        error(f'Error al crear avg_emotion: {e}')

    # 6. Rellenar NaNs en columnas numéricas
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Rellenando NaNs en columnas numéricas para {period}...')
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        success(f'Valores faltantes en columnas numéricas rellenados para {period} ✔')
    except Exception as e:
        error(f'Error al rellenar columnas numéricas: {e}')

    # 7. Alinear columnas con las del entrenamiento
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Alineando columnas con las del entrenamiento para {period}...')
    try:
        missing_cols = [col for col in training_cols if col not in df.columns]
        for col in missing_cols:
            df[col] = 0
        extra_cols = [col for col in df.columns if col not in training_cols]
        df = df.drop(columns=extra_cols)
        df = df[training_cols]
        success(f'Columnas alineadas con las del entrenamiento para {period} ✔')
    except Exception as e:
        error(f'Error al alinear columnas con las del entrenamiento: {e}')

    # Info final
    print('\n' + '.' * 60)
    print(Fore.YELLOW + f'[INFO] Forma final del DataFrame para {period}: {df.shape}')
    print(Fore.YELLOW + f'[INFO] Columnas finales: {df.columns.tolist()}')
    print('.' * 60)
    return df

# Llamada al preprocesamiento
if prediction_period in training_columns and not df.empty:
    df_processed = preprocess_data(df, training_columns[prediction_period], prediction_period)

    # Reasociar id_user con los datos procesados para usarlo después en la predicción
    df_processed['id_user'] = id_users.reset_index(drop=True)
else:
    error(f"No hay columnas de entrenamiento o datos vacíos para el periodo '{prediction_period}'")
    exit()

# -------------------------------
# 5. Generación de Predicciones
# -------------------------------
print('\n' + '=' * 60)
info('Paso 5: Generando predicciones...')
predictions = {}

def generate_predictions(df_input, period, model, scaler, training_columns):
    print(Fore.CYAN + f'[INFO] Procesando predicciones para {period}...')

    if df_input.shape[0] > 0:
        # Asegurarse de eliminar id_user antes de transformar
        features = df_input.drop(columns=['id_user'], errors='ignore')

        if features.shape[0] > 0 and features.shape[1] > 0:
            X_scaled = scaler.transform(features)

            if np.isnan(X_scaled).any():
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            # Generar predicciones y añadir al DataFrame
            predictions[period] = model.predict(X_scaled)
            df_input['predicted_maxAnxietyLevel'] = predictions[period]
            success(f'Predicciones generadas exitosamente: {len(predictions[period])} prediccione(s) obtenidas ✔')

            # Imprimir predicción con su ID
            print(Fore.YELLOW + f'[INFO] Predicciones para formualrios de {period} generadas::')
            for idx, row in df_input.iterrows():
                print(Fore.MAGENTA + f" - Usuario con ID {row['id_user']} - Predicción del nivel de ansiedad: {row['predicted_maxAnxietyLevel']:.4f}")
        else:
            warning(f'No se pueden generar predicciones para {period}: DataFrame vacío o sin columnas procesadas.')
    else:
        warning(f'No se pueden generar predicciones para {period}: DataFrame vacío.')

# Aplicar predicción para night y morning si existen modelos y datos
if 'night' in models:
    generate_predictions(df_processed, 'night', models['night'], scalers['night'], training_columns)

if 'morning' in models:
    generate_predictions(df_processed, 'morning', models['morning'], scalers['morning'], training_columns)

# --------------------------------------
# 6. Guardado y Subida de Predicciones
# --------------------------------------
print('\n' + '=' * 60)
info('Paso 6: Guardando y subiendo predicciones...')

output_dir = './predictions/'
os.makedirs(output_dir, exist_ok=True)

# Función para guardar y subir predicciones
def save_and_upload_predictions(df, predictions):
    global period  # Accedemos a la variable global period
    # Aseguramos que hay datos para el período
    if len(predictions) > 0:
        print('-' * 60)
        print(Fore.CYAN + f'\n[INFO] Guardando predicciones para {period}...')

        # Preparamos el DataFrame con las predicciones y el id_user
        df_output = df[['recorded_at', 'id_user']].copy()
        df_output['predicted_maxAnxietyLevel'] = predictions
        df_output['recorded_at'] = df_output['recorded_at'].apply(lambda x: datetime.now())  # Usamos la fecha actual
        output_path = os.path.join(output_dir, f'predicciones_{period}_anxiety.csv')
        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')

        success(f'Predicciones para {period} guardadas en {output_path} ✔')

        # Guardar las predicciones en el archivo de log
        log_path = os.path.join(output_dir, 'log_de_predicciones.txt')
        with open(log_path, 'a') as log_file:
            for _, row in df_output.iterrows():
                log_file.write(f"id_user: {row['id_user']}, "
                               f"recorded_at: {row['recorded_at'].isoformat()}, "
                               f"prediction: {row['predicted_maxAnxietyLevel']}, "
                               f"model: Linear Regression, "
                               f"form period: {period}\n")
        success(f'Log de predicciones actualizado en {log_path} ✔')

        # Subir las predicciones a Firebase
        print('-' * 60)
        print(Fore.CYAN + f'[INFO] Subiendo predicciones para {period} a Firebase...')
        try:
            for _, row in df_output.iterrows():
                doc_id = f'pred_{period}_{row["recorded_at"].strftime("%Y%m%d_%H%M%S")}'
                db.collection('model_predictions').document(doc_id).set({
                    'id_user': row['id_user'],
                    'recorded_at': row['recorded_at'].isoformat(),
                    'predicted_maxAnxietyLevel': float(row['predicted_maxAnxietyLevel']),
                    'model': 'Linear Regression',
                    'form period': period
                })
            success(f'Predicciones para {period} subidas a Firebase exitosamente ✔')
        except Exception as e:
            error(f'Error al subir predicciones para {period} a Firebase: {e}')
    else:
        warning(f'No se guardaron/subieron predicciones para {period}: datos insuficientes.')

# Establecemos el valor de period y llamamos a la función para cada periodo
if 'night' in predictions:
    period = 'noche'
    save_and_upload_predictions(df, predictions['night'])

if 'morning' in predictions:
    period = 'mañana'
    save_and_upload_predictions(df, predictions['morning'])

# Final del script
print('\n' + '=' * 60)
success('Ejecución del script completada exitosamente ✔')
print('=' * 60)