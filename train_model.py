import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time
import pytz
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from colorama import Fore, Style, init
import os

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
try:
    cred = credentials.Certificate('./credentials_firebase/onehealth-f4967-firebase-adminsdk-fbsvc-e899f7b095.json')
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    success('Conexión a Firebase realizada exitosamente ✔')
except Exception as e:
    error(f'Error al inicializar Firebase: {e}')
    exit()

# ------------------------------------------
# 2. Obtención de Formularios de Firestore
# ------------------------------------------
print('\n' + '=' * 60)
info('Paso 2: Obteniendo datos de la colección formularios...')
try:
    docs = db.collection('formularios').stream()
    data = [doc.to_dict() for doc in docs]
    df = pd.DataFrame(data)
    success(f'{len(df)} formularios encontrados y cargados correctamente ✔')
    print(Fore.YELLOW + f'[INFO] Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas')
    print(Fore.YELLOW + f'[INFO] Columnas en datos crudos: {df.columns.tolist()}')
except Exception as e:
    error(f'Fallo al leer los formularios desde Firestore: {e}')
    exit()

# --------------------------------------------------------------
# 3. Clasificación y Separación en Mañana y Noche de formularios
# --------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 3: Clasificando y separando formularios en "mañana" y "noche"...')
try:
    df['recorded_at'] = pd.to_datetime(df['recorded_at']).dt.tz_convert('Europe/Madrid')
    success('Fechas convertidas y ajustadas a zona horaria Europe/Madrid ✔')

    def classify_period(row):
        hour = row['recorded_at'].hour
        minute = row['recorded_at'].minute
        if time(6, 0) <= time(hour, minute) < time(19, 0):
            return 'mañana'
        else:
            return 'noche'

    df['period'] = df.apply(classify_period, axis=1)
    df_morning = df[df['period'] == 'mañana'].copy()
    df_night = df[df['period'] == 'noche'].copy()

    output_dir = './output/'
    os.makedirs(output_dir, exist_ok=True)
    df_morning.to_csv(os.path.join(output_dir, 'formularios_manana.csv'), index=False, encoding='utf-8-sig')
    df_night.to_csv(os.path.join(output_dir, 'formularios_noche.csv'), index=False, encoding='utf-8-sig')
    success('Formularios clasificados y guardados correctamente ✔')
    print(Fore.YELLOW + f'[INFO] Formularios de mañana: {len(df_morning)}')
    print(Fore.YELLOW + f'[INFO] Formularios de noche: {len(df_night)}')

except Exception as e:
    error(f'Error al clasificar formularios: {e}')
    exit()

# --------------------------
# 4. Procesamiento de Datos
# --------------------------
print('\n' + '=' * 60)
info('Paso 4: Procesando datos...')

def process_data(df,tipo_form, min_rows=10, min_non_null_ratio=0.2):
    print('\n' + '-' * 60)
    print(Fore.LIGHTMAGENTA_EX + f'[INFO] Procesando DataFrame de {tipo_form} con {df.shape[0]} instancias')
    
    # -----------------------------------------------------------------------------
    # 4.1 Eliminar Columnas No Relevantes
    # -----------------------------------------------------------------------------
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Eliminando columnas no relevantes...')
    critical_columns = ['sadnessLevel', 'avgEnergyLevel', 'happinessLevel', 'maxAnxietyLevel']
    columns_to_drop = ['id_user', 'doc_id', 'recorded_at', 'period']
    try:
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        success('Columnas no relevantes eliminadas correctamente ✔')
    except Exception as e:
        error(f'Error al eliminar columnas no relevantes: {e}')
    
    # -----------------------------------------------------------------------------
    # 4.2 Eliminar Filas con Valores Nulos
    # -----------------------------------------------------------------------------
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Eliminando filas con valores nulos...')
    for col in critical_columns:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
    critical_cols_present = [col for col in critical_columns if col in df.columns]
    if critical_cols_present:
        df = df.dropna(subset=critical_cols_present, how='all')
    try:
        print(Fore.YELLOW + f'[INFO] Filas tras eliminar nulos en columnas críticas: {df.shape[0]}')
        success('Filas eliminadas correctamente en columnas críticas ✔')
    except Exception as e:
        error(f'Error al eliminar filas en columnas críticas: {e}')

    # -----------------------------------------------------------------------------
    # 4.3 Convertir Variables Categóricas a Numéricas
    # -----------------------------------------------------------------------------
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Convirtiendo variables categóricas a numéricas mediante one-hot encoding...')
    categorical_cols = ['country', 'state', 'city', 'final_ranking']
    df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], dummy_na=False)
    if not df.empty:
        success('Variables categóricas convertidas a numéricas mediante one-hot encoding ✔')
    else:
        warning('No se encontraron datos para convertir variables categóricas a numéricas.')
    
    # -----------------------------------------------------------------------------
    # 4.4 Detectar y Eliminar Duplicados
    # -----------------------------------------------------------------------------
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Eliminando duplicados...')
    initial_row_count = len(df)
    df.drop_duplicates(inplace=True)
    final_row_count = len(df)
    if final_row_count < initial_row_count:
        success(f'[{tipo_form.upper()}] Duplicados eliminados correctamente ✔ ({initial_row_count - final_row_count} filas eliminadas)')
    else:
        info(f'[{tipo_form.upper()}] No se encontraron duplicados para eliminar.')

    # -----------------------------------------------------------------------------
    # 5.5 Análisis y Manejo de Outliers
    # -----------------------------------------------------------------------------
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Eliminando duplicados...')
    def remove_outliers(df, column):
        if column in df.columns and df[column].notna().sum() > 1:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2 * IQR
            upper_bound = Q3 + 2 * IQR
            initial_count = df.shape[0]
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            removed_count = initial_count - df.shape[0]
            if removed_count > 0:
                print(Fore.YELLOW + f'[INFO] Eliminados {removed_count} outliers en la columna {column}')
            return df
        return df

    numeric_cols = [col for col in critical_columns if col in df.columns]
    for col in numeric_cols:
        df = remove_outliers(df, col)
    success('Procesamiento de Outliers completado correctamente ✔')
    
    # -----------------------------------------------------------------------------
    # 4.5 Crear Nuevas Variables Calculadas
    # -----------------------------------------------------------------------------
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Calculando variables claculadas...')
    # ----------------------------------------------
    # 4.5.1 Calcular Tiempo Total de Redes Sociales 
    # ----------------------------------------------
    print('.' * 60)
    print(Fore.CYAN + f'[INFO] Calculando el tiempo total invertido en redes sociales...')
    if 'instagram_time' in df.columns and 'tiktok_time' in df.columns:
        df['social_media_time'] = df['instagram_time'] + df['tiktok_time']
        df['social_media_time'] = df['social_media_time'].fillna(df['social_media_time'].mean())
        success('Variable social_media_time creada correctamente ✔')
    # --------------------------------------------------------------------
    # 5.5.2 Extraer app más usada, segunda más usada y tercera más usada
    # --------------------------------------------------------------------
    print('.' * 60)
    print(Fore.CYAN + f'[INFO] Extrayendo las aplicaciones más usadas del ranking final...')
    if 'final_ranking' in df.columns:
        def extract_top_apps(ranking_string, position):
            try:
                apps = ranking_string.split(',')
                return apps[position] if len(apps) > position else None
            except Exception:
                return None

        df['top1_app'] = df['final_ranking'].apply(lambda x: extract_top_apps(x, 0))
        df['top2_app'] = df['final_ranking'].apply(lambda x: extract_top_apps(x, 1))
        df['top3_app'] = df['final_ranking'].apply(lambda x: extract_top_apps(x, 2))
        success('Aplicación más usada, segunda más usada y tercera más usada creadas correctamente ✔')
    else:
        error('Columna "final_ranking" no disponible en este tipo de formulario.')

    # ----------------------------------------------
    # 4.5.3 Calcular Promedio de Estado de Ánimo
    # ----------------------------------------------
    print('.' * 60)
    print(Fore.CYAN + f'[INFO] Calcualndo el promedio del estado del animo...')
    mood_cols = ['happinessLevel', 'sadnessLevel', 'apathyLevel', 'avgAnxietyLevel', 'avgEnergyLevel']
    existing_mood_cols = [col for col in mood_cols if col in df.columns]
    if existing_mood_cols:
        df['average_mood'] = df[existing_mood_cols].mean(axis=1)
        success('Variable estado de ánimo promedio creada correctamente ✔')
    else:
        error('No se encontraron columnas necesarias para calcular el estado de ánimo promedio.')

    # ----------------------------------------------
    # 4.5.4 Calcular cantidad de horas de sueño
    # ----------------------------------------------
    print('.' * 60)
    print(Fore.CYAN + f'[INFO] Calculando la duración del sueño...')
    if 'sleep_time' in df.columns and 'wake_up_time' in df.columns:
        try:
            df['sleep_time'] = pd.to_datetime(df['sleep_time'])
            df['wake_up_time'] = pd.to_datetime(df['wake_up_time'])
            df['sleep_duration_hours'] = (df['wake_up_time'] - df['sleep_time']).dt.total_seconds() / 3600
            success('Variable duración del sueño creada correctamente ✔')
        except Exception as e:
            error(f'Error calculando duración del sueño: {e}')
    else:
        error('Columnas necesarias para calcular la duración del sueño no encontradas.')

    # -----------------------------------------------------------------------------
    # 4.6 Rellenar valores faltantes con la media
    # -----------------------------------------------------------------------------
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Rellenando valores faltantes con la media...')
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        success('Valores faltantes en columnas numéricas rellenados con la media ✔')
    else:
        warning('No se encontraron columnas numéricas para rellenar valores faltantes.')
    
    # -----------------------------------------------------------------------------
    # 4.6 Rellenar valores faltantes con la media
    # -----------------------------------------------------------------------------
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Eliminando columnas con baja proporción de valores no nulos o varianza cero...')
    columns_to_remove = []
    for col in df.columns:
        if col == 'maxAnxietyLevel':
            continue
        non_null_ratio = df[col].notna().sum() / df.shape[0]
        if non_null_ratio < min_non_null_ratio:
            print(Fore.YELLOW + f'[INFO] Eliminando columna {col}: proporción de valores no nulos ({non_null_ratio:.2f}) menor que {min_non_null_ratio}')
            columns_to_remove.append(col)
        elif df[col].dtype in ['float64', 'int64'] and df[col].std() == 0:
            print(Fore.YELLOW + f'[INFO] Eliminando columna {col}: varianza cero')
            columns_to_remove.append(col)
    df = df.drop(columns=columns_to_remove)
    success('Columnas con baja proporción de valores no nulos o varianza cero eliminadas ✔')
    
    # -----------------------------------------------------------------------------
    # 4.6 Mostrar dataframe resultante del procesamiento de los datos
    # -----------------------------------------------------------------------------
    print('-' * 60)
    print(Fore.CYAN + f'[INFO] Mostrando DataFrame resultante del procesamiento...')
    if df.shape[0] < min_rows:
        warning(f'El DataFrame tiene solo {df.shape[0]} fila(s), menos que el mínimo requerido ({min_rows}).')
    if df.shape[1] == 0:
        warning('El DataFrame no tiene columnas después del preprocesamiento.')
    
    print(Fore.YELLOW + f'[INFO] Forma final del DataFrame: {df.shape}')
    print(Fore.YELLOW + f'[INFO] Columnas finales: {df.columns.tolist()}')
    return df

df_morning = process_data(df_morning, "mañana")
df_night = process_data(df_night, "noche")
success('\nProcesamiento de datos completado para ambos períodos ✔\n')

# --------------------------------------
# 5. Análisis Exploratorio de Datos
# --------------------------------------
print('\n' + '=' * 60)
print(Fore.BLUE + f'Paso 5: Realizando análisis exploratorio de datos...')

def perform_eda(df, title):
    print('\n' + '-' * 60)
    print(Fore.LIGHTMAGENTA_EX + f'[INFO] Realizando EDA para {title}...')
    
    if df.empty or df.shape[1] == 0:
        warning(f'No se puede realizar el EDA para {title}: DataFrame vacío o sin columnas.')
        return
    
    # ----------------------------------------------------------------------------------------
    # 7.1 Estadísticas Descriptiva del DataFrame
    # ----------------------------------------------------------------------------------------
    print(Fore.CYAN + f'[INFO] Obteniendo estadísticas descriptiva del dataframe...')
    try:
        print(df.describe())
    except ValueError as e:
        error(f'No se puede generar estadísticas descriptivas para {title}: {e}')
        return
    
    # ----------------------------------------------------------------------------------------
    # 7.1 Matriz de Correlación
    # ----------------------------------------------------------------------------------------
    print(Fore.CYAN + f'[INFO][{title.upper()}] Generando matriz de correlación...')
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(method='pearson', min_periods=1)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, mask=np.isnan(corr))
        plt.title(f'Matriz de Correlación - {title}')
        plt.savefig(f'./output/correlation_matrix_{title.lower().replace(" ", "_")}.png')
        plt.close()
        success(f'Matriz de correlación generada y guardada para {title} ✔')
    else:
        warning(f'No hay suficientes columnas numéricas para generar la matriz de correlación en {title}.')

    # ----------------------------------------------------------------------------------------
    # 7.2 Histograma
    # ---------------------------------------------------------------------------------------
    print(Fore.CYAN + f'[INFO][{title.upper()}] Generando histogramas...')
    for col in numeric_cols:
        if df[col].dropna().shape[0] > 0:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, bins=20)
            plt.title(f'Histograma de {col} - {title}')
            plt.xlabel(col)
            plt.ylabel('Frecuencia')
            plt.savefig(f'./output/histogram_{col}_{title.lower().replace(" ", "_")}.png')
            plt.close()
            success(f'Histograma de {col} generado y guardado para {title} ✔')
        else:
            warning(f'No se puede generar histograma para {col} en {title} debido a datos insuficientes.')
    # ----------------------------------------------------------------------------------------
    # 7.3 Boxplots
    # ----------------------------------------------------------------------------------------
    print(Fore.CYAN + f'[INFO][{title.upper()}] Generando boxplots...')
    for col in numeric_cols:
        if df[col].dropna().shape[0] > 1:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col].dropna())
            plt.title(f'Boxplot de {col} - {title}')
            plt.xlabel(col)
            plt.savefig(f'./output/boxplot_{col}_{title.lower().replace(" ", "_")}.png')
            plt.close()
            success(f'Boxplot de {col} generado y guardado para {title} ✔')
        else:
            warning(f'No se puede generar boxplot para {col} en {title} debido a datos insuficientes.')
    
    # ----------------------------------------------------------------------------------------
    # 7.4 Scatter Plots
    # ----------------------------------------------------------------------------------------
    print(Fore.CYAN + f'[INFO][{title.upper()}] Generando scatter plots...')
    if 'social_media_time' in df.columns and 'maxAnxietyLevel' in df.columns:
        if df['social_media_time'].dropna().shape[0] > 0 and df['maxAnxietyLevel'].dropna().shape[0] > 0:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x='social_media_time', y='maxAnxietyLevel')
            plt.title(f'Relación entre social_media_time y maxAnxietyLevel - {title}')
            plt.xlabel('social_media_time')
            plt.ylabel('maxAnxietyLevel')
            plt.savefig(f'./output/scatter_social_media_anxiety_{title.lower().replace(" ", "_")}.png')
            plt.close()
            success(f'Scatter plot de social_media_time vs maxAnxietyLevel generado y guardado para {title} ✔')
        else:
            warning(f'No se puede generar scatter plot para social_media_time y maxAnxietyLevel en {title} debido a datos insuficientes.')

perform_eda(df_morning, 'Formularios de Mañana')
perform_eda(df_night, 'Formularios de Noche')
success('\nAnálisis exploratorio de datos completado ✔\n')

# --------------------------------------
# 6. Preparación de Datos para Modelado
# --------------------------------------
print('\n' + '=' * 60)
info('Paso 6: Preparando datos para entrenar...')

def prepare_data(df,form_type, target='maxAnxietyLevel', min_rows=10):
    print(Fore.CYAN + '[INFO] Dividiendo dataframe {form_typeen} conjuntos de entrenamiento y prueba...')
    if df.shape[0] < min_rows:
        error(f'El DataFrame tiene solo {df.shape[0]} fila(s). No se puede modelar.')
        return None
    if df.shape[1] == 0:
        error('El DataFrame no tiene columnas para modelar.')
        return None
    if target not in df.columns:
        error(f'La columna objetivo {target} no está en el DataFrame.')
        return None
    
    X = df.select_dtypes(include=['number']).drop(columns=[target], errors='ignore')
    y = df[target]
    
    if X.shape[1] == 0:
        error('No hay columnas numéricas suficientes para modelar.')
        return None
    
    X = X.fillna(X.mean())
    
    zero_variance_cols = X.columns[X.std() == 0]
    if len(zero_variance_cols) > 0:
        print(Fore.YELLOW + f'[INFO] Eliminando columnas con varianza cero: {list(zero_variance_cols)}')
    X = X.drop(columns=zero_variance_cols)
    
    if X.shape[1] == 0:
        error('No hay columnas válidas después de eliminar varianza cero.')
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
        warning('Hay valores NaN en los datos escalados. Rellenando con ceros.')
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
    
    success(f'División exitosa: Conjunto de entrenamiento (X_train): {X_train.shape}, Conjunto de prueba (X_test): {X_test.shape} ✔')
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

result_morning = prepare_data(df_morning, "mañana")
if result_morning is None:
    warning('No se puede modelar para Morning debido a datos insuficientes o columna objetivo faltante.')
    X_morning, X_morning_test, X_morning_train_scaled, X_morning_test_scaled, y_morning_train, y_morning_test, scaler_morning = None, None, None, None, None, None, None
else:
    X_morning, X_morning_test, X_morning_train_scaled, X_morning_test_scaled, y_morning_train, y_morning_test, scaler_morning = result_morning
    success('Datos para Morning preparados correctamente ✔')

result_night = prepare_data(df_night, "noche")
if result_night is None:
    warning('No se puede modelar para Night debido a datos insuficientes o columna objetivo faltante.')
    X_night, X_night_test, X_night_train_scaled, X_night_test_scaled, y_night_train, y_night_test, scaler_night = None, None, None, None, None, None, None
else:
    X_night, X_night_test, X_night_train_scaled, X_night_test_scaled, y_night_train, y_night_test, scaler_night = result_night
    success('Datos para Night preparados correctamente ✔')

# --------------------------
# 7. Entrenamiento de Modelos
# --------------------------
print('\n' + '=' * 60)
info('Paso 7: Entrenando modelos...')

def train_and_evaluate(X_train, X_train_scaled, X_test_scaled, y_train, y_test, name):
    print(Fore.LIGHTMAGENTA_EX + f'\n[INFO] Entrenando modelos para {name}...')
    if X_train_scaled is None or X_test_scaled is None or y_train is None:
        error(f'No se pueden entrenar modelos para {name} debido a datos insuficientes.')
        return None
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    results = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {'MSE': mse, 'R2': r2}
        print(Fore.YELLOW + f'[INFO] {name} - {model_name} - MSE: {mse:.2f}, R2: {r2:.2f}')
    success(f'Modelos entrenados y evaluados para {name} ✔')
    return results, X_train.columns

morning_results, morning_columns = train_and_evaluate(X_morning, X_morning_train_scaled, X_morning_test_scaled, y_morning_train, y_morning_test, 'Mañana')
night_results, night_columns = train_and_evaluate(X_night, X_night_train_scaled, X_night_test_scaled, y_night_train, y_night_test, 'Noche')

# ---------------------------------------------------------------
# 8. Guardar Modelos y Subir Resultados de validación a Firebase
# ---------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 8: Validando modelos y subiendo resultados a Firebase...')

if X_night_train_scaled is not None and y_night_train is not None:
    print(Fore.CYAN + '\n[INFO] Validando modelo para formualrios de noche...')
    best_model = RandomForestRegressor(random_state=42)
    best_model.fit(X_night_train_scaled, y_night_train)
    y_pred = best_model.predict(X_night_test_scaled)
    mse = mean_squared_error(y_night_test, y_pred)
    r2 = r2_score(y_night_test, y_pred)
    
    print(Fore.YELLOW + f'[INFO] Resultados para predicciones de formularios de noche con RandomForestRegressor:')
    print(Fore.YELLOW + f'  MSE: {mse:.2f}')
    print(Fore.YELLOW + f'  R²: {r2:.2f}')
    
    os.makedirs('modelos_guardados', exist_ok=True)
    joblib.dump(best_model, 'modelos_guardados/model_night_anxiety.pkl')
    joblib.dump(night_columns, 'modelos_guardados/training_columns_night_anxiety.pkl')
    joblib.dump(scaler_night, 'modelos_guardados/scaler_night_anxiety.pkl')
    success('Modelo, columnas y scaler guardados exitosamente para noche ✔')
    
    results = {
        'night': {
            'model': 'Random Forest',
            'mse': mse,
            'r2': r2
        }
    }
    try:
        db.collection('resultados').document('modelo_noche_anxiety').set(results)
        success('Resultados subidos a Firebase exitosamente para noche ✔')
    except Exception as e:
        error(f'Error al subir resultados a Firebase para noche: {e}')
else:
    warning('No se pudo guardar el modelo para noche debido a datos insuficientes.')

if X_morning_train_scaled is not None and y_morning_train is not None:
    print(Fore.CYAN + '\n[INFO] Validando modelo para formualrios de mañana...')
    best_model = RandomForestRegressor(random_state=42)
    best_model.fit(X_morning_train_scaled, y_morning_train)
    y_pred = best_model.predict(X_morning_test_scaled)
    mse = mean_squared_error(y_morning_test, y_pred)
    r2 = r2_score(y_morning_test, y_pred)
    
    print(Fore.YELLOW + f'[INFO] Resultados para predicciones de formularios de mañana con RandomForestRegressor:')
    print(Fore.YELLOW + f'  MSE: {mse:.2f}')
    print(Fore.YELLOW + f'  R²: {r2:.2f}')
    
    os.makedirs('modelos_guardados', exist_ok=True)
    joblib.dump(best_model, 'modelos_guardados/model_morning_anxiety.pkl')
    joblib.dump(morning_columns, 'modelos_guardados/training_columns_morning_anxiety.pkl')
    joblib.dump(scaler_morning, 'modelos_guardados/scaler_morning_anxiety.pkl')
    success('Modelo, columnas y scaler guardados exitosamente para mañana ✔')
    
    results = {
        'morning': {
            'model': 'Random Forest',
            'mse': mse,
            'r2': r2
        }
    }
    try:
        db.collection('resultados').document('modelo_morning_anxiety').set(results)
        success('Resultados subidos a Firebase exitosamente para mañana ✔')
    except Exception as e:
        error(f'Error al subir resultados a Firebase para mañana: {e}')
else:
    warning('No se pudo guardar el modelo para mañana debido a datos insuficientes.')

print('\n' + '=' * 60)
success('Ejecución del script completada exitosamente ✔')
print('=' * 60)