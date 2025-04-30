# ONEHEALTH_INTELLIGENT
Mediante la aplicación de técnicas de aprendizaje automático, esta herramienta permitirá predecir posibles ataques de ansiedad basándose en datos recopilados de lasdiferentes fuentes:

- Formularios rellenados por el usuario a través de la aplicación.
- Datos biométricos obtenidos de dispositivos wearables.
- Datos del dispositivo y su uso por parte del usuario mediante la API de Capacitor integrada en la aplicación.

## Instalación 

Antes de comenzar, asegúrate de cumplir con los siguientes requisitos:

1. **Instalar Python**: Descarga e instala Python desde su [sitio oficial](https://www.python.org/). Asegúrate de agregar Python al PATH durante la instalación.

2. **Instalar dependencias**: Ejecuta el siguiente comando para instalar las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```

3. **Obtener credenciales de Firebase**: Sigue estos pasos para configurar Firebase:
    - Accede a **Firebase**.
    - Ve a **Configuración del proyecto**.
    - Dirígete a **Cuentas de servicio**.
    - Haz clic en **Generar nueva clave privada**.
    - Guarda la clave generada en la carpeta `/Credentials_firebase` y asegúrate de importarla correctamente desde el código.
