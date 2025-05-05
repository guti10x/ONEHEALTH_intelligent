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
# 8. Preparación de Datos para Modelado
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 8: Preparando los datos para entrenamiento y prueba...')

from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------------------
# 8.1 División de df_morning
# ----------------------------------------------------------------------------------------
print(Fore.CYAN + '[INFO] Verificando columnas y dimensiones de df_morning...')
print("Dimensiones de df_morning:", df_morning.shape)
print("Columnas de df_morning:", df_morning.columns)

numeric_cols_morning = df_morning.select_dtypes(include=['number']).columns

if len(numeric_cols_morning) <= 1 and 'happinessLevel' in numeric_cols_morning:
    print(Fore.RED + "[ERROR] No hay suficientes columnas numéricas en df_morning para modelar.")
    X_morning_train, X_morning_test, y_morning_train, y_morning_test = None, None, None, None
else:
    X_morning = df_morning.select_dtypes(include=['number']).drop(columns=['happinessLevel'], errors='ignore')
    y_morning = df_morning['happinessLevel'] if 'happinessLevel' in df_morning.columns else None

    print(Fore.CYAN + "Dimensiones de X_morning:", X_morning.shape)
    print(Fore.CYAN + "Dimensiones de y_morning:", y_morning.shape)

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
        print(Fore.GREEN + '[ÉXITO] División de df_morning realizada con éxito:')
        print(" - X_morning_train:", X_morning_train.shape)
        print(" - X_morning_test :", X_morning_test.shape)

# ----------------------------------------------------------------------------------------
# 8.2 División de df_night
# ----------------------------------------------------------------------------------------
print(Fore.CYAN + '\n[INFO] Verificando columnas y dimensiones de df_night...')
print("Dimensiones de df_night:", df_night.shape)
print("Columnas de df_night:", df_night.columns)

numeric_cols_night = df_night.select_dtypes(include=['number']).columns

if len(numeric_cols_night) <= 1 and 'happinessLevel' in numeric_cols_night:
    print(Fore.RED + "[ERROR] No hay suficientes columnas numéricas en df_night para modelar.")
    X_night_train, X_night_test, y_night_train, y_night_test = None, None, None, None
else:
    X_night = df_night.select_dtypes(include=['number']).drop(columns=['happinessLevel'], errors='ignore')
    y_night = df_night['happinessLevel'] if 'happinessLevel' in df_night.columns else None

    print(Fore.CYAN + "Dimensiones de X_night:", X_night.shape)
    print(Fore.CYAN + "Dimensiones de y_night:", y_night.shape)

    if X_night.shape[0] <= 1:
        print(Fore.RED + "[ERROR] df_night tiene solo una fila o está vacío. No se puede dividir.")
        X_night_train, X_night_test, y_night_train, y_night_test = None, None, None, None
    else:
        print(Fore.CYAN + '[INFO] Rellenando valores nulos en X_night con la media...')
        X_night = X_night.fillna(X_night.mean())

        print(Fore.CYAN + '[INFO] Dividiendo df_night en conjuntos de entrenamiento y prueba...')
        X_night_train, X_night_test, y_night_train, y_night_test = train_test_split(
            X_night, y_night, test_size=0.2, random_state=42
        )
        print(Fore.GREEN + '[ÉXITO] División de df_night realizada con éxito:')
        print(" - X_night_train:", X_night_train.shape)
        print(" - X_night_test :", X_night_test.shape)

# ----------------------------------------------------------------------------------------
# 9 Escalado de datos numéricos
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 8.3: Escalando variables numéricas...')

# Escalado para df_morning
if X_morning_train is not None and X_morning_test is not None:
    print(Fore.CYAN + '[INFO] Aplicando escalado a df_morning...')
    scaler_morning = StandardScaler()
    X_morning_train_scaled = scaler_morning.fit_transform(X_morning_train)
    X_morning_test_scaled = scaler_morning.transform(X_morning_test)
    print(Fore.GREEN + '[ÉXITO] Escalado de df_morning completado.')
else:
    X_morning_train_scaled, X_morning_test_scaled = None, None
    print(Fore.YELLOW + '[ADVERTENCIA] No se realizó el escalado de df_morning debido a datos insuficientes o división fallida.')

# Escalado para df_night
if X_night_train is not None and X_night_test is not None:
    print(Fore.CYAN + '[INFO] Aplicando escalado a df_night...')
    scaler_night = StandardScaler()
    X_night_train_scaled = scaler_night.fit_transform(X_night_train)
    X_night_test_scaled = scaler_night.transform(X_night_test)
    print(Fore.GREEN + '[ÉXITO] Escalado de df_night completado.')
else:
    X_night_train_scaled, X_night_test_scaled = None, None
    print(Fore.YELLOW + '[ADVERTENCIA] No se realizó el escalado de df_night debido a datos insuficientes o división fallida.')

# ----------------------------------------------------------------------------------------
# 9. Entrenamiento y Evaluación de Modelos (df_night)
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 9: Entrenando y evaluando modelos de regresión con df_night...')

# Validación de datos
if X_night is not None and y_night is not None and X_night.shape[0] > 1:
    print(Fore.CYAN + '[INFO] Dividiendo nuevamente df_night para entrenamiento y prueba (80/20)...')
    X_night_train, X_night_test, y_night_train, y_night_test = train_test_split(X_night, y_night, test_size=0.2, random_state=42)

    print(Fore.CYAN + '[INFO] Limpiando columnas vacías o no numéricas...')
    X_night_train = X_night_train.dropna(axis=1, how='all')
    X_night_test = X_night_test.dropna(axis=1, how='all')
    X_night_train = X_night_train.fillna(X_night_train.mean())
    X_night_test = X_night_test.fillna(X_night_test.mean())
    X_night_train = X_night_train.select_dtypes(include=['number'])
    X_night_test = X_night_test.select_dtypes(include=['number'])

    print(Fore.CYAN + '[INFO] Eliminando columnas con varianza cero...')
    zero_var_cols = X_night_train.columns[X_night_train.std() == 0]
    X_night_train = X_night_train.drop(columns=zero_var_cols)
    X_night_test = X_night_test.drop(columns=zero_var_cols)

    print(Fore.CYAN + '[INFO] Escalando los datos...')
    scaler = StandardScaler()
    X_night_train_scaled = scaler.fit_transform(X_night_train)
    X_night_test_scaled = scaler.transform(X_night_test)

    if np.isnan(X_night_train_scaled).any():
        print(Fore.RED + '[ERROR] Se encontraron NaN en los datos escalados. Revisa el preprocesamiento.')
    else:
        print(Fore.GREEN + '[ÉXITO] Datos escalados correctamente. Entrenando modelos...')

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

        # Entrenando y evaluando modelos
        for name, model in models.items():
            model.fit(X_night_train_scaled, y_night_train)
            y_pred = model.predict(X_night_test_scaled)
            mse = mean_squared_error(y_night_test, y_pred)
            r2 = r2_score(y_night_test, y_pred)
            print(Fore.YELLOW + f'{name} - MSE: {mse:.2f}, R²: {r2:.2f}')

            # ----------------------------------------------------------------------------------------
            # 9.1 Gráfica de comparación: Valores reales vs. predichos
            # ----------------------------------------------------------------------------------------
            print(Fore.CYAN + f'[INFO] Generando gráfico de Reales vs. Predichos para {name}...')
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_night_test, y=y_pred)
            plt.plot([y_night_test.min(), y_night_test.max()], [y_night_test.min(), y_night_test.max()], 'r--')
            plt.xlabel('Valores Reales')
            plt.ylabel('Valores Predichos')
            plt.title(f'Comparación: Reales vs. Predichos ({name})')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # ----------------------------------------------------------------------------------------
            # 9.2 Guardar modelo entrenado
            # ----------------------------------------------------------------------------------------
            print(Fore.CYAN + f'[INFO] Guardando el modelo {name} entrenado...')
            import joblib
            import os

            # Crear carpeta si no existe
            os.makedirs('modelos_guardados', exist_ok=True)

            # Guardar modelo
            ruta_modelo = f'modelos_guardados/{name.lower().replace(" ", "_")}_night.pkl'
            joblib.dump(model, ruta_modelo)
            print(Fore.CYAN + f'[INFO] Modelo {name} guardado exitosamente en: {ruta_modelo}')

            # ----------------------------------------------------------------------------------------
            # 9.3 Predicciones con nuevos datos
            # ----------------------------------------------------------------------------------------
            def predecir_nuevos_datos(nuevo_df, modelo):
                """
                Aplica el modelo entrenado para predecir valores de felicidad en nuevos datos nocturnos.
                """
                try:
                    print(Fore.CYAN + '[INFO] Preparando nuevos datos para predicción...')
                    # Procesamiento similar al entrenamiento
                    nuevo_df = nuevo_df.select_dtypes(include=['number'])
                    nuevo_df = nuevo_df.fillna(nuevo_df.mean())

                    # Alinear columnas con las del entrenamiento
                    nuevo_df = nuevo_df[X_night_train.columns]

                    # Escalar
                    nuevo_df_scaled = scaler.transform(nuevo_df)

                    # Predecir
                    predicciones = modelo.predict(nuevo_df_scaled)
                    print(Fore.CYAN + '[INFO] Predicción realizada con éxito.')
                    return predicciones

                except Exception as e:
                    print(Fore.RED + f'[ERROR] No se pudo hacer la predicción: {e}')
                    return None

else:
    print(Fore.RED + '[ERROR] df_night no tiene suficientes datos válidos para entrenar modelos.')


# ----------------------------------------------------------------------------------------
# 10. Validación y Subida a Firebase (df_night)
# ----------------------------------------------------------------------------------------
print('\n' + '=' * 60)
info('Paso 9.X: Validando el modelo y subiendo los resultados a Firebase...')

# Verificar que los datos necesarios estén definidos
if 'X_night_train_scaled' in locals() and 'y_night_train' in locals() and \
   'X_night_test_scaled' in locals() and 'y_night_test' in locals():
    
    # Definir y entrenar el modelo con todos los datos escalados
    best_model = RandomForestRegressor(random_state=42)
    best_model.fit(X_night_train_scaled, y_night_train)
    
    # Realizar predicciones sobre el conjunto de prueba
    y_pred = best_model.predict(X_night_test_scaled)
    
    # Calcular métricas de rendimiento
    mse = mean_squared_error(y_night_test, y_pred)
    r2 = r2_score(y_night_test, y_pred)
    
    # Mostrar resultados en la consola para depuración
    print(Fore.GREEN + f"Resultados para 'noche' con RandomForestRegressor:")
    print(Fore.GREEN + f"  MSE: {mse:.2f}")
    print(Fore.GREEN + f"  R²: {r2:.2f}")
    
    # Crear un diccionario con los resultados
    results = {
        'night': {
            'model': 'Random Forest',
            'mse': mse,
            'r2': r2
        }
    }
    
    # Subir los resultados a Firebase con manejo de excepciones
    try:
        db.collection('resultados').document('modelo_noche').set(results)
        print(Fore.GREEN + "Resultados subidos a Firebase exitosamente.")
    except Exception as e:
        print(Fore.RED + f"Error al subir resultados a Firebase: {e}")
else:
    print(Fore.RED + "[ERROR] Los datos de entrenamiento o prueba no están definidos.")

