import firebase_admin
from firebase_admin import credentials, firestore
from colorama import Fore, Style, init
import joblib
import os
from datetime import datetime, timedelta
import pandas as pd

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
    'scaler': 'modelos_guardados/scaler_mañana.pkl'
}

morning_files = {
    'model': 'modelos_guardados/model_mañana_Linear_Regression.pkl',
    'columns': 'modelos_guardados/training_columns_mañana.pkl',
    'scaler': 'modelos_guardados/scaler_noche.pkl'
}

# Cargar solo los archivos del periodo actual de predicción
if prediction_period == 'night':
    # Cargar el modelo de "noche"
    if os.path.exists(night_files['model']):
        models['night'] = joblib.load(night_files['model'])
        success('Modelo de noche cargado exitosamente ✔')
    else:
        warning('No se encontró el archivo del modelo de noche. No se podrá realizar la predicción para noche.')

    # Cargar las columnas de entrenamiento de "noche"
    if os.path.exists(night_files['columns']):
        training_columns['night'] = joblib.load(night_files['columns'])
        success('Columnas de entrenamiento de noche cargadas exitosamente ✔')
    else:
        warning('No se encontraron las columnas de entrenamiento de noche. No se podrá realizar la predicción para noche.')

    # Cargar el escalador de "noche"
    if os.path.exists(night_files['scaler']):
        scalers['night'] = joblib.load(night_files['scaler'])
        success('Escalador de noche cargado exitosamente ✔')
    else:
        warning('No se encontró el escalador de noche. No se podrá realizar la predicción para noche.')

elif prediction_period == 'morning':
    # Cargar el modelo de "mañana"
    if os.path.exists(morning_files['model']):
        models['morning'] = joblib.load(morning_files['model'])
        success('Modelo de mañana cargado exitosamente ✔')
    else:
        warning('No se encontró el archivo del modelo de mañana. No se podrá realizar la predicción para mañana.')

    # Cargar las columnas de entrenamiento de "mañana"
    if os.path.exists(morning_files['columns']):
        training_columns['morning'] = joblib.load(morning_files['columns'])
        success('Columnas de entrenamiento de mañana cargadas exitosamente ✔')
    else:
        warning('No se encontraron las columnas de entrenamiento de mañana. No se podrá realizar la predicción para mañana.')

    # Cargar el escalador de "mañana"
    if os.path.exists(morning_files['scaler']):
        scalers['morning'] = joblib.load(morning_files['scaler'])
        success('Escalador de mañana cargado exitosamente ✔')
    else:
        warning('No se encontró el escalador de mañana. No se podrá realizar la predicción para mañana.')


