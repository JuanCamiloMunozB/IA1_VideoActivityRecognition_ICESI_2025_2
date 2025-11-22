## Manual de Usuario: Video Activity Recognition

---

## 1. Descripción general

El sistema de **Video Activity Recognition** permite reconocer actividades humanas en tiempo real usando la cámara web del equipo.  
La aplicación:

- Captura video desde la webcam.
- Detecta la pose con MediaPipe.
- Calcula métricas posturales (inclinación del tronco y ángulos de rodilla).
- Clasifica la actividad actual con un modelo SVM entrenado previamente.
- Muestra la predicción y las métricas directamente sobre el video.

La interfaz se ejecuta en una ventana de OpenCV llamada:

> `HAR Tiempo Real - SVM`

---

## 2. Requisitos del sistema

### 2.1. Hardware

- Cámara web funcional (integrada o USB).
- CPU con soporte para operaciones de punto flotante (cualquier equipo moderno).
- Se recomienda contar con al menos 8 GB de RAM para un funcionamiento fluido.

### 2.2. Windows (ejecutable)

- Sistema operativo: Windows 10 o superior (64 bits).
- No es necesario tener Python instalado para usar el **ejecutable**.
- Permisos para ejecutar aplicaciones descargadas y acceso a la cámara.

### 2.3. Linux / Mac (Docker)

- Docker instalado y funcionando.
- Acceso al dispositivo de video, normalmente `/dev/video0`.
- Para entornos gráficos X11: permisos de acceso al servidor gráfico (`xhost`).

---

## 3. Formas de ejecución

El sistema se puede ejecutar de dos maneras:

1. **Ejecutable para Windows** (`VideoActivityRecognition.exe`).
2. **Imagen Docker** para Linux/Mac (o también en Windows con Docker Desktop + X11).

---

## 4. Ejecución en Windows (ejecutable)

### 4.1. Descarga

Descargar el ejecutable desde:

```
https://drive.google.com/file/d/1FdMknTccJYTd1WjFiD7QRudbJ90btPzD/view?usp=sharing

```

Guárdalo en una carpeta de tu preferencia, por ejemplo: C:\VideoActivityRecognition\.


### 4.2. Formas de ejecutar

#### ✔ Opción A - Doble clic

- Navega a la carpeta donde descargaste el ejecutable.
- Haz doble clic en **VideoActivityRecognition.exe**.
- La cámara se activará y aparecerá la ventana principal del sistema.

#### ✔ Opción B - Línea de comandos

- Abre **CMD** o **PowerShell**.
- Cambia al directorio donde está el ejecutable:  
    cd C:\\VideoActivityRecognition  

- Ejecuta:  
    .\\VideoActivityRecognition.exe  

**Nota:** El ejecutable utiliza internamente app_entry.py como punto de entrada, configura el entorno del modelo y lanza la aplicación gráfica en tiempo real.

## 5\. Ejecución con Docker (Linux / Mac)

### 5.1. Descargar la imagen Docker

Descargar el archivo .tar desde el siguiente enlace:

Descargar Imagen Docker (Google Drive)

### 5.2. Cargar la imagen

docker load -i video-har.tar  

### 5.3. Habilitar acceso gráfico (X11)

xhost +local:docker  

### 5.4. Ejecutar el contenedor

Verifica que tu cámara sea /dev/video0. Luego inicia el contenedor:

sudo docker run -it --rm \\  
\--device=/dev/video0:/dev/video0 \\  
\-e DISPLAY=\$DISPLAY \\  
\-v /tmp/.X11-unix:/tmp/.X11-unix:rw \\  
\--network host \\  
video-activity-recognition:latest  

Esto ejecutará automáticamente la aplicación en tiempo real con la ventana de OpenCV.

## 6\. Uso de la aplicación

### 6.1. Inicio del sistema

Al iniciar la aplicación se mostrará en consola un mensaje como:

\============================================================  
Video Activity Recognition - Real-time HAR System  
\============================================================  
Presiona 'q' en la ventana de video para salir  

- Se abrirá la ventana **HAR Tiempo Real - SVM**.
- La cámara se activará automáticamente.

### 6.2. Elementos en pantalla

#### 📌 Panel de información (arriba izquierda)

Incluye:

- Modelo cargado (SVM Full o Reduced)
- FPS estimado
- Visibilidad media de landmarks
- Advertencias de baja visibilidad

**Ejemplo visual:**

Modelo: SVM reduced | FPS: 30.0  
Visibilidad media: 0.85  

#### Estado de la actividad

- **Antes de tener suficientes frames:**  
    Actividad: --- (calentando ventana)  

- **Cuando el sistema ya puede predecir:**  
    Actividad: walking_forward (92.3%)  

**Código de colores según confianza:**

- 🔴 **Rojo:** probabilidad < 40%
- 🟡 **Amarillo:** 40% - 70%
- 🟢 **Verde:** > 70%

#### Métricas posturales (panel secundario)

Incluye valores calculados en posture_metrics.py:

Metricas postura:  
trunk_inclination_deg: 4.3  
knee_angle_l_deg: 91.7  
knee_angle_r_deg: 89.2  

#### Mensaje de salida

Abajo de la ventana verás:

Pulsa 'q' para salir  

## 7\. Cómo salir de la aplicación

- En la ventana de video, presiona la tecla **q**.
- La cámara se liberará y la ventana se cerrará.
- **En Docker:** El contenedor se elimina automáticamente gracias al flag --rm.

## 8\. Solución de problemas

### 8.1. Windows bloquea el ejecutable

**Mensaje:** _"Windows protegió tu PC"_

- **Solución:**
  - Clic en **Más información**.
  - Clic en **Ejecutar de todas formas**.

### 8.2. Error: no se puede abrir la cámara

**Consola muestra:**

\[UI\] No se pudo abrir la camara con indice 0  

**Soluciones:**

- Cerrar otras apps que usan cámara (Zoom, Teams, Meet, etc.).
- Revisar permisos: _Configuración → Privacidad → Cámara → Activar acceso_.
- Si tienes varias cámaras, puede que el índice correcto no sea 0.

### 8.3. Advertencia de baja visibilidad

**Si aparece:**

Advertencia: baja visibilidad  

**Causas probables:**

- Hay poca iluminación.
- Estás muy lejos de la cámara.
- El fondo está saturado o hay oclusiones.

**Solución:** Acercarse, mejorar la luz o cambiar el ángulo.


## 9\. Notas técnicas

- La predicción usa un **SVM optimizado** (versión _reduced_ por defecto).
- El predictor acumula frames en un _buffer_ y usa un muestreo configurable (frame_sample_every).
- Los cálculos de _trunk inclination_ y _knee angles_ salen del módulo posture_metrics.py.
- El sistema usa **MediaPipe Pose** con:
  - min_detection_confidence=0.5
  - min_tracking_confidence=0.5

