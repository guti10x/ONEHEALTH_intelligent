import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate('./credentials_firebase/onehealth-f4967-firebase-adminsdk-fbsvc-e899f7b095.json')

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

print('Conexión a Firebase completada correctamente')
