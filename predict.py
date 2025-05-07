import firebase_admin
from firebase_admin import credentials, firestore
from colorama import Fore, Style, init

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