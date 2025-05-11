import firebase_admin
from firebase_admin import credentials, firestore
from colorama import Fore, Style, init
import joblib
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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
print(Fore.BLUE + '[INFO] Iniciando el script de predicción...')

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
        print(Fore.YELLOW + f'[INFO] Obteniendo formularios de la mañana...')
    else:
        prediction_period = 'morning'  # Estamos en noche, predecimos para mañana
        print(Fore.YELLOW + f'[INFO] Obteniendo formularios de la noche...')

    # Determinar el rango de tiempo en función del periodo de predicción
    if prediction_period == 'night':
        start_time = datetime.combine(today - timedelta(days=1), datetime.min.time()) + timedelta(hours=19)
        end_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=6)
    else:
        start_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=6)
        end_time = datetime.combine(today, datetime.min.time()) + timedelta(hours=19)

    period = prediction_period  # Conservamos esta variable para compatibilidad

    info(f'Obteniendo formularios entre: {start_time} y {end_time} ({period})')

    # Filtrar formularios en el rango de tiempo
    docs = db.collection('formularios').where('recorded_at', '>=', start_time).where('recorded_at', '<', end_time).stream()
    data = [doc.to_dict() for doc in docs]
    df = pd.DataFrame(data)
    # Imprimir los datos obtenidos de Firebase
    print(Fore.CYAN + '[DEBUG] Datos obtenidos de Firebase:')
    for i, doc in enumerate(data):
        print(f"Documento {i + 1}: {doc}")
    success(f'{len(df)} formularios encontrados y cargados correctamente ✔')
    print(Fore.YELLOW + f'[INFO] Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas')
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
    'model': 'modelos_guardados/model_noche_Linear_Regression.pkl',
    'columns': 'modelos_guardados/training_columns_noche.pkl',
    'scaler': 'modelos_guardados/scaler_noche.pkl'
}

morning_files = {
    'model': 'modelos_guardados/model_mañana_Linear_Regression.pkl',
    'columns': 'modelos_guardados/training_columns_mañana.pkl',
    'scaler': 'modelos_guardados/scaler_mañana.pkl'
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

# Definición de la función de preprocesamiento de datos
def preprocess_data(df, training_cols, period):
    print(Fore.CYAN + f'\n[INFO] Preprocesando datos para {period} con {df.shape[0]} filas iniciales')

    critical_columns = ['sadnessLevel', 'avgEnergyLevel', 'happinessLevel']
    print(Fore.YELLOW + f'[INFO] Columnas críticas presentes: {[col for col in critical_columns if col in df.columns]}')

    columns_to_drop = ['id_user', 'doc_id', 'recorded_at', 'period', 'maxAnxietyLevel']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    success(f'Columnas no relevantes eliminadas para {period} ✔')

    for col in critical_columns:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
    success(f'Valores faltantes en columnas críticas rellenados para {period} ✔')

    categorical_cols = ['country', 'state', 'city', 'final_ranking']
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], dummy_na=False)
    success(f'Variables categóricas convertidas a numéricas para {period} ✔')

    if 'instagram_time' in df.columns and 'tiktok_time' in df.columns:
        df['social_media_time'] = df['instagram_time'] + df['tiktok_time']
        df['social_media_time'] = df['social_media_time'].fillna(df['social_media_time'].mean())
        success(f'Variable social_media_time creada para {period} ✔')

    if 'sadnessLevel' in df.columns and 'happinessLevel' in df.columns:
        df['avg_emotion'] = (df['sadnessLevel'] + df['happinessLevel']) / 2
        df['avg_emotion'] = df['avg_emotion'].fillna(df['avg_emotion'].mean())
        success(f'Variable avg_emotion creada para {period} ✔')

    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    success(f'Valores faltantes en columnas numéricas rellenados para {period} ✔')

    missing_cols = [col for col in training_cols if col not in df.columns]
    for col in missing_cols:
        df[col] = 0

    extra_cols = [col for col in df.columns if col not in training_cols]
    df = df.drop(columns=extra_cols)
    df = df[training_cols]
    success(f'Columnas alineadas con las del entrenamiento para {period} ✔')

    print(Fore.YELLOW + f'[INFO] Forma final del DataFrame para {period}: {df.shape}')
    print(Fore.YELLOW + f'[INFO] Columnas finales: {df.columns.tolist()}')
    return df

# Llamada al preprocesamiento en este paso
if prediction_period in training_columns and not df.empty:
    df_processed = preprocess_data(df, training_columns[prediction_period], prediction_period)
else:
    error(f"No hay columnas de entrenamiento o datos vacíos para el periodo '{prediction_period}'")
    exit()

# -------------------------- 
# 6. Generación de Predicciones
# --------------------------
print('\n' + '=' * 60)
info('Paso 6: Generando predicciones...')
predictions = {}

def generate_predictions(df_input, period, model, scaler, training_columns):
    print(Fore.CYAN + f'\n[INFO] Procesando predicciones para {period}...')
    
    if df_input.shape[0] > 0:
        
        if df_processed.shape[0] > 0 and df_processed.shape[1] > 0:
            X_scaled = scaler.transform(df_processed)

            if np.isnan(X_scaled).any():
                X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            predictions[period] = model.predict(X_scaled)
            df_input['predicted_maxAnxietyLevel'] = predictions[period]
            success(f'Predicciones generadas para {period}: {len(predictions[period])} registros ✔')
            
            # Mostrar las predicciones generadas
            print(Fore.GREEN + f'[INFO] Predicciones para formualrio de {period}: {df_input["predicted_maxAnxietyLevel"].tolist()}')
        else:
            warning(f'No se pueden generar predicciones para {period}: DataFrame vacío o sin columnas procesadas.')
    else:
        warning(f'No se pueden generar predicciones para {period}: DataFrame vacío.')

# Aplicar predicción para night y morning si existen modelos y datos
if 'night' in models:
    generate_predictions(df, 'night', models['night'], scalers['night'], training_columns)

if 'morning' in models:
    generate_predictions(df, 'morning', models['morning'], scalers['morning'], training_columns)
