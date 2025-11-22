# Ejecutable de Reconocimiento de Actividades - Video Activity Recognition

## 🎯 Inicio Rápido

### Ejecutar la Aplicación

```bash
cd dist
.\VideoActivityRecognition.exe
```

O simplemente haz **doble clic** en `VideoActivityRecognition.exe` en la carpeta `dist/`.

### Controles

- **Iniciar**: El reconocimiento comienza automáticamente al abrir
- **Salir**: Presiona la tecla `q` en la ventana de video

## 📦 Lo que se Incluye

El ejecutable **NO requiere** instalación de Python ni dependencias. Todo está incluido:

✅ Python runtime  
✅ OpenCV (cámara y procesamiento de video)  
✅ MediaPipe (detección de pose)  
✅ Modelos de Machine Learning (SVM)  
✅ Datos de entrenamiento (features.csv)  
✅ Todo el código de Entrega2 y Entrega3  

## 🔨 Reconstruir el Ejecutable

Si actualizas los modelos o el código y necesitas regenerar el `.exe`:

> **⚠️ Importante**: Debes activar el entorno virtual `.venv` antes de ejecutar el build.

```bash
# Desde la raíz del proyecto

# 1. Activa el entorno virtual
.venv\Scripts\Activate.ps1

# 2. Ejecuta el build
python build_exe.py
```

El script automáticamente:
1. Valida que todos los archivos necesarios existan
2. Instala PyInstaller si no está instalado
3. Borra builds anteriores
4. Genera un nuevo `VideoActivityRecognition.exe` en `dist/`

## 📋 Archivos del Proyecto

```
IA1_VideoActivityRecognition_ICESI_2025_2/
├── app_entry.py              # Punto de entrada del ejecutable
├── build_exe.py              # Script de construcción automática
├── dist/
│   └── VideoActivityRecognition.exe  # ← EJECUTABLE FINAL (7.47 MB)
├── Entrega2/
│   ├── src/                  # Código de feature engineering
│   └── experiments/
│       └── results/
│           └── features.csv  # Requerido
└── Entrega3/
    ├── src/                  # Código de UI e inferencia
    └── experiments/
        └── models/           # Modelos .joblib
```

## ⚠️ Solución de Problemas

### "Windows protegió tu PC"

Este mensaje aparece porque el ejecutable no está firmado digitalmente.

**Solución:**
1. Haz clic en "Más información"
2. Haz clic en "Ejecutar de todas formas"

### No se puede acceder a la cámara

**Solución:**
1. Ve a **Configuración de Windows** → **Privacidad** → **Cámara**
2. Asegúrate de que "Permitir que las aplicaciones accedan a tu cámara" esté **activado**
3. Cierra otras aplicaciones que usen la cámara (Zoom, Teams, etc.)

### El ejecutable no inicia o se cierra inmediatamente

**Solución:**
1. Ejecuta desde la línea de comandos para ver los errores:
   ```bash
   cd dist
   .\VideoActivityRecognition.exe
   ```
2. Reconstruye el ejecutable:
   ```bash
   python build_exe.py
   ```

## 🚀 Distribución

Puedes compartir el archivo `VideoActivityRecognition.exe` con cualquier persona que tenga Windows.

**Requisitos del destinatario:**
- Windows 10 u 11
- Cámara web
- ~500 MB de espacio en disco (el ejecutable es ~243 MB)

**NO necesitan:**
- Python instalado
- Ninguna dependencia adicional
- Archivos de código fuente

## 📊 Detalles Técnicos

- **Tamaño**: 242.88 MB
- **Herramienta**: PyInstaller 6.x
- **Modo**: Single file executable (console mode para debugging)
- **Plataforma**: Windows x64
- **Dependencias empaquetadas**: OpenCV 4.12.0, MediaPipe, scikit-learn, XGBoost, pandas, numpy, scipy

---

**Creado el**: 21/11/2025  
**Versión**: 1.0
