# Entrega 3 – Optimización y Despliegue en Tiempo Real

Esta entrega cubre: (1) reducción de características usando SVM, (2) optimización del modelo, (3) pruebas de despliegue y (4) interfaz en tiempo real para reconocimiento de actividad humana (HAR).

## Objetivos

- Optimizar el modelo SVM mediante selección de características para reducir dimensionalidad y mejorar eficiencia.
- Implementar un sistema de inferencia en tiempo real.
- Desarrollar una interfaz de usuario para demostración del HAR.

## Cómo Usar

### Requisitos Previos

- Python 3.8 o superior
- Entorno virtual activado
- Dependencias instaladas (ver abajo)

### Instalación

1. Crear y activar entorno virtual (desde la raíz del proyecto):

   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

2. Instalar dependencias:

   ```bash
   cd Entrega3
   pip install -r requirements.txt
   ```

### Ejecución

1. **Reducción de Características:**
   - Ejecuta el notebook `notebooks/01_svm_feature_reduction.ipynb` para seleccionar las mejores características usando SVM.

2. **Pruebas de Despliegue:**
   - Ejecuta el notebook `notebooks/02_deployment_tests.ipynb` para verificar la carga de modelos y predicciones.

3. **Demo en Tiempo Real:**
   - Ejecuta la aplicación de interfaz:
     ```bash
     python src/online/ui_app.py
     ```
   - Esto iniciará una interfaz web para probar el reconocimiento en tiempo real usando la cámara.

## Estructura del Directorio

```
Entrega3/
├── docs/
│   └── manual_usuario.md          # Manual de usuario (pendiente)
├── experiments/
│   ├── logs/                      # Logs de experimentos
│   ├── models/                    # Modelos entrenados (svm_full.joblib, svm_reduced.joblib, label_encoder.joblib)
│   └── results/                   # Resultados (feature_reduction_summary.md, selected_features.json)
├── notebooks/
│   ├── 01_svm_feature_reduction.ipynb  # Reducción de características
│   └── 02_deployment_tests.ipynb       # Pruebas de despliegue
├── src/
│   ├── models/
│   │   └── load_artifacts.py      # Carga de modelos y artefactos
│   ├── online/
│   │   ├── posture_metrics.py     # Cálculo de métricas posturales
│   │   ├── realtime_inference.py  # Inferencia en tiempo real
│   │   └── ui_app.py              # Aplicación de interfaz web
│   └── utils/
│       ├── config.py              # Configuraciones
│       └── preprocessing.py       # Funciones de preprocesamiento
└── requirements.txt               # Dependencias de Python
```

## Dependencias

- streamlit (para la interfaz web)
- scikit-learn
- numpy
- pandas
- joblib
- opencv-python
- mediapipe (si se usa para inferencia en tiempo real)

Ver `requirements.txt` para la lista completa.

## Resultados Esperados

- Modelo SVM optimizado con menos características pero rendimiento similar.
- Interfaz funcional para HAR en tiempo real.
- Logs y resúmenes de los experimentos en `experiments/`.

