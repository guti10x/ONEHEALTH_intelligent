import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

### 1- Conexión con la base de datos de Firebase #############################################################################
print('-' * 60)
print('CONECTANDO A FIREBASE...')
# Inicializar Firebase
cred = credentials.Certificate('./credentials_firebase/onehealth-f4967-firebase-adminsdk-fbsvc-e899f7b095.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

print('Conexión a Firebase realizada exitosamente ✔')

### 2- Obtención de todos los datos recopilados ###############################################################################
print('-' * 60)
print('OBTENIENDO DATOS DE FIREBASE...')
# De Formularios
formularios = [] 
try:
    doc_ref = db.collection('formularios')
    docs = doc_ref.stream()

    for doc in docs:
        formularios.append({'id': doc.id, 'data': doc.to_dict()})

    print(f'{len(formularios)} formularios encontrados y cargados correctamente ✔')
except Exception as e:
    print('Error leyendo Firestore:', e)

# De datos biométricos
biometric_data = [] 
# Pending

### 3- PREPROCESADO DE DATOS ###########################################################################################
print('-' * 60)
print('PREPROCESANDO DATOS...')        

# Convertimos la lista de formularios en un DataFrame
formularios_data = [f['data'] for f in formularios]
df = pd.DataFrame(formularios_data)

# 3.1 Eliminamos columnas No relevantes
print('Eliminando columnas NO relevantes...')
columns_to_drop = ['id_user']
columns_to_remove = [col for col in columns_to_drop if col in df.columns]
if columns_to_remove:
    print(f'Columnas no relevantes eliminadas: {columns_to_remove} ✔\n')
    df = df.drop(columns=columns_to_remove)

# 3.2 Eliminar filas con valores nulos
print('Eliminando filas con valores nulos...')
nulos = df[df.isnull().any(axis=1)]
if not nulos.empty:
    print(f'Se han eliminado correctamente {len(nulos)} filas con valores nulos detectadas ✔\n')
    df = df.dropna()
else:
    print('No se han encontrado filas con valores nulos ✔\n')

# 3.3 Convertir variables categóricas (texto) a números
print('Convirtiendo variables categóricas a variables numéricas...')
df = pd.get_dummies(df)
print('Variables categóricas convertidas a variables numéricas ✔\n')

print('-' * 60)
