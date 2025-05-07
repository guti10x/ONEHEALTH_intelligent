import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta, time
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def save_dataframe(df, filename, label):
    try:
        output_dir = './output/'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        success(f'DataFrame "{label}" guardado exitosamente en {filepath} ✔')
        print(Fore.YELLOW + f'[INFO] Número de instancias guardadas de formularios de {label}": {len(df)}')
    except Exception as e:
        error(f'Error al guardar DataFrame "{label}": {e}')

def check_non_null_columns(df, nombre_df):
    null_counts = df.isnull().sum()
    valid_columns = null_counts[null_counts < len(df)].index.tolist()
    if valid_columns:
        print(Fore.YELLOW + f'[INFO] Columnas con al menos un valor no nulo en {nombre_df}:')
        for col in valid_columns:
            print(f'   - {col} ({len(df) - null_counts[col]} valores no nulos)')
    else:
        print(Fore.RED + f'[ALERTA] Todas las columnas en {nombre_df} están completamente vacías.')

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
       :!JYYYYYYYYYYJJYYYYJJJJ?!^                      OneHealth Intelligent V1 - Train                                           
         .^!7?JJYYYYYYYYJJ?7~:.                                                                    
             ..:^^~~~~^::.                                                                                                                                                                                                                                         
""")
print("-" * 100)
print(Fore.BLUE + '[INFO] Iniciando el script de entrenamiento...')

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
    #print(f'Formularios: {formularios}')
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
    print(Fore.CYAN + '\n[INFO] Ajustando zona horaria de recorded_at...')
    try:
        df['recorded_at'] = pd.to_datetime(df['recorded_at']).dt.tz_convert('Europe/Madrid')
        success('Fechas convertidas y ajustadas a zona horaria Europe/Madrid ✔')
    except Exception as e:
        error(f'Error al convertir las fechas: {e}')

    # Función para clasificar el período
    print(Fore.CYAN + '\n[INFO] Clasificando formularios en "mañana" y "noche"...')
    def classify_period(row):
        if 'happinessLevel' in row and pd.notna(row['happinessLevel']):
            return 'noche'
        else:
            return 'mañana'

    # Aplicar clasificación
    df['period'] = df.apply(classify_period, axis=1)

    # Separar en dos DataFrames
    df_morning = df[df['period'] == 'mañana'].copy()
    df_night = df[df['period'] == 'noche'].copy()

    # Mostrar conteo
    print(Fore.YELLOW + f'[INFO] Formularios de mañana: {len(df_morning)}')
    print(Fore.YELLOW + f'[INFO] Formularios de noche: {len(df_night)}')

    if not df_morning.empty and not df_night.empty:
        success('Formularios clasificados y DataFrames creados correctamente ✔')
    else:
        warning('Algunos de los DataFrames "mañana" o "noche" están vacíos.')
else:
    warning('No se encontraron formularios para procesar.')

# Verificar que hay suficientes datos no nulos por columna
print(Fore.CYAN + '\n[INFO] Verificando atributos con datos no nulos...')

check_non_null_columns(df_morning, 'df_morning')
check_non_null_columns(df_night, 'df_night')


# ----------------------------------------------------------------------------------------
# 5. Procesamiento de datos:
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 5: Procesando datos...')

for df_label, df_ref in [('mañana', df_morning), ('noche', df_night)]:
    print('\n' + '-' * 100)
    print(Fore.BLUE + f'\n[INFO] Procesando formularios de: {df_label.upper()}')

    # -----------------------------------------------------------------------------
    # 5.1 Eliminar Columnas No Relevantes
    # -----------------------------------------------------------------------------
    print(Fore.CYAN + f'\n[INFO] Eliminando columnas no relevantes...')
    columns_to_drop = ['id_user', 'recorded_at', 'period']
    df_ref.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    if all(col not in df_ref.columns for col in columns_to_drop):
        success(f'[{df_label.upper()}] Columnas no relevantes eliminadas correctamente ✔')
    else:
        warning(f'[{df_label.upper()}] Algunas columnas no relevantes no se pudieron eliminar.')

    # -----------------------------------------------------------------------------
    # 5.2 Eliminar Filas con Valores Nulos
    # -----------------------------------------------------------------------------
    print(Fore.CYAN + f'\n[INFO] Eliminando filas con valores nulos...')
    critical_columns = ['sadnessLevel', 'happinessLevel', 'avgEnergyLevel']
    initial_row_count = len(df_ref)
    df_ref.dropna(subset=critical_columns, inplace=True)
    final_row_count = len(df_ref)
    if final_row_count < initial_row_count:
        success(f'[{df_label.upper()}] Filas con valores nulos eliminadas correctamente ✔ ({initial_row_count - final_row_count} filas eliminadas)')
    else:
        info(f'[{df_label.upper()}] No se encontraron filas con valores nulos en las columnas críticas.')


    # -----------------------------------------------------------------------------
    # 5.3 Convertir Variables Categóricas a Numéricas
    # -----------------------------------------------------------------------------
    print(Fore.CYAN + f'\n[INFO] Convirtiendo variables categóricas a numéricas mediante one-hot encoding...')
    try:
        df_ref = pd.get_dummies(df_ref, columns=['country', 'state', 'city'], drop_first=True)
        success(f'[{df_label.upper()}] Variables categóricas convertidas correctamente a numéricas ✔')
    except Exception as e:
        error(f'[{df_label.upper()}] Error al convertir variables categóricas: {e}')

    # -----------------------------------------------------------------------------
    # 5.4 Detectar y Eliminar Duplicados
    # -----------------------------------------------------------------------------
    print(Fore.CYAN + f'\n[INFO] Eliminando duplicados en el DataFrame {df_label.upper()}...')
    initial_row_count = len(df_ref)
    df_ref.drop_duplicates(inplace=True)
    final_row_count = len(df_ref)
    if final_row_count < initial_row_count:
        success(f'[{df_label.upper()}] Duplicados eliminados correctamente ✔ ({initial_row_count - final_row_count} filas eliminadas)')
    else:
        info(f'[{df_label.upper()}] No se encontraron duplicados para eliminar.')

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

    print(Fore.CYAN + f'\n[INFO] Eliminado Outliers {df_label.upper()}...')
    numeric_cols = ['sadnessLevel', 'avgEnergyLevel', 'maxAnxietyLevel', 'happinessLevel']
    for col in numeric_cols:
        if col in df_ref.columns:
            initial_row_count = len(df_ref)
            df_ref = remove_outliers(df_ref, col)
            final_row_count = len(df_ref)
            removed_count = initial_row_count - final_row_count
            success(f'[{df_label.upper()}] Outliers eliminados en columna "{col}". Filas eliminadas: {removed_count}')
            success(f'[{df_label.upper()}] Outliers procesados correctamente para columna "{col}" ✔')


    # -----------------------------------------------------------------------------
    # 5.5 Crear Nuevas Variables Calculadas
    # -----------------------------------------------------------------------------

    # ----------------------------------------------
    # 5.5.1 Calcular Tiempo Total de Redes Sociales 
    # ----------------------------------------------
    print(Fore.CYAN + f'\n[INFO] Calculando el tiempo total invertido en redes sociales...')
    if 'instagram_time' in df_ref.columns and 'tiktok_time' in df_ref.columns:
        df_ref['total_social_media_time'] = df_ref['instagram_time'] + df_ref['tiktok_time']
        success(f'[{df_label.upper()}] Variable del tiempo total invertido en redes sociales creada correctamente ✔')
    else:
        error(f'[{df_label.upper()}] Columnas necesarias para calcular el tiempo total de redes sociales no encontradas.')

    # ----------------------------------------------
    # 5.5.2 Extraer app más usada, segunda más usada y tercera más usada
    # ----------------------------------------------
    if 'final_ranking' in df_ref.columns:
        print(Fore.CYAN + f'\n[INFO] Extrayendo las aplicaciones más usadas del ranking final...')

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
        print(Fore.CYAN + f'\n[INFO] Columnas utilizadas para el cálculo: {existing_mood_cols}')
        df_ref['average_mood'] = df_ref[existing_mood_cols].mean(axis=1)
        success(f'[{df_label.upper()}] Variable estado de ánimo promedio creada correctamente ✔')
    else:
        error(f'[{df_label.upper()}] No se encontraron columnas necesarias para calcular el estado de ánimo promedio.')

    # ----------------------------------------------
    # 5.5.4 Calcular cantidad de horas de sueño
    # ----------------------------------------------
    print(Fore.CYAN + f'\n[INFO] Calculando la duración del sueño...')
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
info('Paso 6: Guardando DataFrames en archivos CSV y mostrando número de instancias...')

# Guardar DataFrames y mostrar número de instancias
save_dataframe(df_morning, 'df_morning.csv', 'mañana')
save_dataframe(df_night, 'df_night.csv', 'noche')

# ----------------------------------------------------------------------------------------
# 7. Análisis Exploratorio de Datos y Visualización
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 7: Realizando análisis exploratorio de datos y visualización...')

# ----------------------------------------------------------------------------------------
# 7.1 Matriz de Correlación
# ----------------------------------------------------------------------------------------
#Identificamos columnas numéricas en los DataFrames
numeric_columns_morning = df_morning.select_dtypes(include=['number']).columns
numeric_columns_night = df_night.select_dtypes(include=['number']).columns

# df_morning
print(Fore.CYAN + '[INFO][MAÑANA] Visualizando matriz de correlación para df_morning...')
morning_corr = df_morning[numeric_columns_morning].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(morning_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Visualizando Matriz de Correlación - Mañana")
plt.show()

# df_night
print(Fore.CYAN + '[INFO][NOCHE] Visualizando matriz de correlación para df_night...')
night_corr = df_night[numeric_columns_night].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(night_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Visualizando Matriz de Correlación - Noche")
plt.show()

# ----------------------------------------------------------------------------------------
# 7.2 Histogramas
# ---------------------------------------------------------------------------------------
def plot_histograms(df, title):
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].dropna().shape[0] > 0:
            print(Fore.CYAN + f'[INFO][{title.upper()}] VISUALIZANDO HISTOGRAMA PARA {col.upper()}...')
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col], kde=True, bins=20)
            plt.title(f'Histograma de {col} - {title}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.show()
        else:
            print(Fore.YELLOW + f'[ADVERTENCIA][{title.upper()}] No se puede generar histograma para {col} en {title} debido a datos insuficientes.')

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
            print(Fore.CYAN + f'[INFO][{title.upper()}] Visualizando boxplot para {col}...')
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot de {col} - {title}')
            plt.xlabel(col)
            plt.show()
        else:
            print(Fore.YELLOW + f'[ADVERTENCIA][{title.upper()}] No se puede generar boxplot para {col} en {title} debido a datos insuficientes.')

# Aplicación de boxplots
plot_boxplots(df_morning, "Mañana")
plot_boxplots(df_night, "Noche")

# ----------------------------------------------------------------------------------------
# 7.4 Scatter Plots
# ----------------------------------------------------------------------------------------
def plot_scatter(df, x_col, y_col, title):
    if df[x_col].dropna().shape[0] > 0 and df[y_col].dropna().shape[0] > 0:
        print(Fore.CYAN + f'[INFO][{title.upper()}] Visualizando scatter plot entre {x_col} y {y_col}...')
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.title(f'Relación entre {x_col} y {y_col} - {title}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()
    else:
        print(Fore.YELLOW + f'[ADVERTENCIA][{title.upper()}] No se puede generar scatter plot para {x_col} y {y_col} en {title} debido a datos insuficientes.')

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
        print(Fore.CYAN + f'[INFO][{title.upper()}] Generando pairplot para múltiples variables numéricas...')
        sns.pairplot(df[numeric_cols])
        plt.suptitle(f'Pairplot - {title}', y=1.02)
        plt.show()
    else:
        print(Fore.YELLOW + f'[ADVERTENCIA][{title.upper()}] No se puede generar pairplot para {title} debido a columnas insuficientes.')

plot_pairplot(df_morning, "Mañana")
plot_pairplot(df_night, "Noche")

# ----------------------------------------------------------------------------------------
# 8. Preparación de Datos para Modelado
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 8: Preparando los datos para entrenamiento y prueba...')

from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------------
# 8.1 División de df_morning en conjuntos de entrenamiento y prueba
# ----------------------------------------------------------------------------------------
print('\n' + '-' * 100)
print(Fore.CYAN + '[INFO] Verificando columnas y dimensiones de df_morning...')
print(Fore.YELLOW + '[INFO] Dimensiones de df_morning:', df_morning.shape)
print(Fore.YELLOW + '[INFO] Columnas de df_morning:', df_morning.columns)

numeric_cols_morning = df_morning.select_dtypes(include=['number']).columns

if len(numeric_cols_morning) <= 1 and 'happinessLevel' in numeric_cols_morning:
    print(Fore.RED + "[ERROR] No hay suficientes columnas numéricas en df_morning para modelar.")
    X_morning_train, X_morning_test, y_morning_train, y_morning_test = None, None, None, None
else:
    X_morning = df_morning.select_dtypes(include=['number']).drop(columns=['happinessLevel'], errors='ignore')
    y_morning = df_morning['happinessLevel'] if 'happinessLevel' in df_morning.columns else None

    if X_morning.shape[0] <= 1:
        print(Fore.RED + "[ERROR] df_morning tiene solo una fila o está vacío. No se puede dividir.")
        X_morning_train, X_morning_test, y_morning_train, y_morning_test = None, None, None, None
    else:
        print(Fore.CYAN + '[INFO] Rellenando valores nulos en X_morning con la media...')
        X_morning = X_morning.fillna(X_morning.mean())

        print(Fore.CYAN + '[INFO] Dividiendo df_morning en conjuntos de entrenamiento y prueba...')
        X_morning_train, X_morning_test, y_morning_train, y_morning_test = train_test_split(
            X_morning, y_morning, test_size=0.2, random_state=42
        )
        print(Fore.GREEN + '[SUCCES] División de df_morning realizada con éxito:')
        print(Fore.YELLOW + f"[INFO] Tamaño del conjunto de entrenamiento: {X_morning_train.shape}")
        print(Fore.YELLOW + f"[INFO] Tamaño del conjunto de prueba: {X_morning_test.shape}")

        # Guardar DataFrames de entrenamiento y test
        print(Fore.CYAN + '[INFO] Guardando conjuntos de entrenamiento y prueba en archivos CSV...')
        save_dataframe(X_morning_train, 'X_morning_train.csv', 'mañana')
        save_dataframe(X_morning_test, 'X_morning_test.csv', 'noche')
        print(Fore.GREEN + '[SUCCESS] Conjuntos de entrenamiento y prueba guardados exitosamente.')

# ----------------------------------------------------------------------------------------
# 8.2 División de df_night
# ----------------------------------------------------------------------------------------
print('\n' + '-' * 100)
print(Fore.CYAN + '[INFO] Verificando columnas y dimensiones de df_night...')
print(Fore.YELLOW + "Dimensiones de df_night:", df_night.shape)
print(Fore.YELLOW + "Columnas de df_night:", df_night.columns)

numeric_cols_night = df_night.select_dtypes(include=['number']).columns

if len(numeric_cols_night) <= 1 and 'happinessLevel' in numeric_cols_night:
    print(Fore.RED + "[ERROR] No hay suficientes columnas numéricas en df_night para modelar.")
    X_night_train, X_night_test, y_night_train, y_night_test = None, None, None, None
else:
    X_night = df_night.select_dtypes(include=['number']).drop(columns=['happinessLevel'], errors='ignore')
    y_night = df_night['happinessLevel'] if 'happinessLevel' in df_night.columns else None

    if X_night.shape[0] <= 1:
        print(Fore.RED + "[ERROR] df_night tiene solo una fila o está vacío. No se puede dividir.")
        X_night_train, X_night_test, y_night_train, y_night_test = None, None, None, None
    else:
        print(Fore.CYAN + '[INFO] Rellenando valores nulos en X_night con la media...')
        X_night = X_night.fillna(X_night.mean())

        print(Fore.CYAN + '[INFO] Dividiendo df_night en conjuntos de entrenamiento y prueba...')
        X_night_train, X_night_test, y_night_train, y_night_test = train_test_split(X_night, y_night, test_size=0.2, random_state=42)
        print(Fore.GREEN + '[SUCCESS] División de df_night realizada con éxito:')
        print(Fore.YELLOW + "[INFO] Tamaño del conjunto de entrenamiento:", X_night_train.shape)
        print(Fore.YELLOW + "[INFO] Tamaño del conjunto de prueba :", X_night_test.shape)

        # Guardar DataFrames de entrenamiento y test
        print(Fore.CYAN + '[INFO] Guardando conjuntos de entrenamiento y prueba en archivos CSV...')
        save_dataframe(X_night_train, 'X_night_train.csv', 'mañana')
        save_dataframe(X_night_test, 'X_night_test.csv', 'noche')
        print(Fore.GREEN + '[SUCCESS] Conjuntos de entrenamiento y prueba guardados exitosamente.')

# ----------------------------------------------------------------------------------------
# 9. Entrenamiento de Modelos
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 9: Entrenando modelos con formularios de noche y mañana...')

# Función para entrenar y evaluar modelos
def train_and_evaluate_models(X_train, X_test, y_train, y_test, label):
    print(Fore.CYAN + f'\n[INFO] Entrenando modelos para formularios de {label}...')

    print(Fore.CYAN + '\n[INFO] Limpiando columnas vacías o no numéricas...')
    X_train = X_train.dropna(axis=1, how='all')
    X_test = X_test.dropna(axis=1, how='all')
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])
    success('Limpieza de columnas vacías o no numéricas realizada correctamente.')

    print(Fore.CYAN + '\n[INFO] Eliminando columnas con varianza cero...')
    zero_var_cols = X_train.columns[X_train.std() == 0]
    X_train = X_train.drop(columns=zero_var_cols)
    X_test = X_test.drop(columns=zero_var_cols)
    if zero_var_cols.empty:
        print(Fore.YELLOW + '[INFO] No se encontraron columnas con varianza cero para eliminar.')
    else:
        success(f'Se eliminaron columnas con varianza cero: {list(zero_var_cols)} ✔')

    print(Fore.CYAN + '\n[INFO] Escalando los datos...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if np.isnan(X_train_scaled).any():
        print(Fore.RED + '[ERROR] Se encontraron NaN en los datos escalados. Revisa el preprocesamiento.')
    else:
        print(Fore.GREEN + '[SUCCESS] Datos escalados correctamente. Entrenando modelos...')

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

        # Entrenando y evaluando modelos
        print(Fore.CYAN + '\n[INFO] Entrenando modelo...')
        for name, model in models.items():
            print('' + '-' * 60)
            print(Fore.LIGHTMAGENTA_EX + f'[INFO] Entrenando con el modelo {name} para {label}...')
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            success(f'Modelo entrenado correctamente para {label}')
            print(Fore.YELLOW + f' - MSE: {mse:.2f}')
            print(Fore.YELLOW + f' - R²: {r2:.2f}')

            # ----------------------------------------------------------------------------------------
            # Gráfica de comparación: Valores reales vs. predichos
            # ----------------------------------------------------------------------------------------
            print(Fore.CYAN + f'[INFO] Visualizando gráfico de datos reales vs datos Predichos para {name} ({label})...')
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=y_pred)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Valores Reales')
            plt.ylabel('Valores Predichos')
            plt.title(f'Comparación: Reales vs. Predichos ({name} - {label})')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # ----------------------------------------------------------------------------------------
            # Guardar modelo entrenado
            # ----------------------------------------------------------------------------------------
            print(Fore.CYAN + f'[INFO] Guardando el modelo {name} entrenado para {label}...')

            # Crear carpeta si no existe
            os.makedirs('modelos_guardados', exist_ok=True)

            # Guardar modelo
            ruta_modelo = f'modelos_guardados/{name.lower().replace(" ", "_")}_{label.lower()}.pkl'
            joblib.dump(model, ruta_modelo)
            success('[SUCCESS] Modelo {name} guardado exitosamente en: {ruta_modelo}')

# Entrenar modelos para "noche"
if X_night_train is not None and y_night_train is not None and X_night_train.shape[0] > 1:
    train_and_evaluate_models(X_night_train, X_night_test, y_night_train, y_night_test, "Noche")
else:
    print(Fore.RED + '\n[ERROR] df_night no tiene suficientes datos válidos para entrenar modelos.')

# Entrenar modelos para "mañana"
if X_morning_train is not None and y_morning_train is not None and X_morning_train.shape[0] > 1:
    train_and_evaluate_models(X_morning_train, X_morning_test, y_morning_train, y_morning_test, "Mañana")
else:
    print(Fore.RED + '\n[ERROR] df_morning no tiene suficientes datos válidos para entrenar modelos.')


# ----------------------------------------------------------------------------------------
# 10. Validación de los modelos
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 10: Validando los modelos generados...')

def validate_models(X_test, y_test, label):
    print(Fore.LIGHTRED_EX + f'\n[INFO] Validando modelos para {label}...')

    # Escalar los datos de prueba
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    # Cargar modelos guardados
    model_dir = 'modelos_guardados'
    if not os.path.exists(model_dir):
        print(Fore.RED + f'[ERROR] No se encontró el directorio "{model_dir}" con los modelos guardados.')
        return

    for model_file in os.listdir(model_dir):
        if label.lower() in model_file:
            model_path = os.path.join(model_dir, model_file)
            print(Fore.CYAN + f'\n[INFO] Cargando modelo desde: {model_path}...')
            try:
                model = joblib.load(model_path)
                y_pred = model.predict(X_test_scaled)

                # Calcular métricas
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                success(f'Modelo validado correctamente: {model_file}')
                print(Fore.YELLOW + f' - MSE: {mse:.2f}')
                print(Fore.YELLOW + f' - R²: {r2:.2f}')

                # Gráfica de comparación: Valores reales vs. predichos
                print(Fore.CYAN + f'\n[INFO] Generando gráfico de Reales vs. Predichos para {model_file}...')
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=y_pred)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel('Valores Reales')
                plt.ylabel('Valores Predichos')
                plt.title(f'Validación: Reales vs. Predichos ({model_file})')
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            except Exception as e:
                error(f'Error al cargar o validar el modelo {model_file}: {e}')

# Validar modelos para "noche"
if X_night_test is not None and y_night_test is not None and X_night_test.shape[0] > 1:
    validate_models(X_night_test, y_night_test, "Noche")
else:
    print(Fore.RED + '\n[ERROR] No hay suficientes datos válidos en df_night para validar modelos.')

# Validar modelos para "mañana"
if X_morning_test is not None and y_morning_test is not None and X_morning_test.shape[0] > 1:
    validate_models(X_morning_test, y_morning_test, "Mañana")
else:
    print(Fore.RED + '\n[ERROR] No hay suficientes datos válidos en df_morning para validar modelos.')
