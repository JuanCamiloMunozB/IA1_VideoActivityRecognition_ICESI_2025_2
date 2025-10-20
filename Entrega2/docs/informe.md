# Informe – Entrega 2  
## 1. Flujo de trabajo y metodología

El sistema implementado aborda el reconocimiento de actividades humanas (HAR) a partir de videos, 
pero **sin procesar los videos directamente en píxeles**.  
En lugar de ello, se extraen **landmarks corporales** usando *MediaPipe Pose*, 
obteniendo las coordenadas tridimensionales de las articulaciones por frame 
(`x, y, z, visibility`).  
Esto permite representar el movimiento de una persona como una secuencia de posiciones 
numéricas en el espacio, eliminando dependencias de fondo, color o iluminación.

### 1.1. Razonamiento metodológico
No se entrenaron redes convolucionales sobre imágenes o frames, 
porque ese enfoque requiere grandes volúmenes de datos y GPU de alto rendimiento.  
El uso de landmarks ofrece una representación cinemática compacta y eficiente 
que conserva la información relevante para clasificar posturas y movimientos.  
Cada video se convierte así en una **serie temporal de articulaciones humanas**, 
base para el modelado con algoritmos clásicos de machine learning.

### 1.2. Flujo técnico general

El proyecto se organiza en módulos bajo `src/` y se ejecuta mediante tres notebooks:

| Notebook | Etapa | Descripción |
|-----------|--------|-------------|
| `01_preprocesamiento.ipynb` | Limpieza y normalización | Carga datos desde Supabase/CSV, filtra por visibilidad, normaliza y genera `features.csv`. |
| `02_modelado.ipynb` | Entrenamiento | Ejecuta los clasificadores definidos en `src/models/train_models.py` con GridSearchCV. |
| `03_resultados.ipynb` | Evaluación y análisis | Carga métricas, matrices y gráficas de importancia para comparar modelos. |

Los notebooks actúan como controladores de análisis, 
mientras que los archivos de `src/` contienen la lógica modular reutilizable.

---

## 2. Diseño de los modelos

Tres modelos fueron implementados para comparar desempeño:

1. **SVM (Support Vector Machine)**  
   - Implementado con `sklearn.svm.SVC` en un pipeline con `StandardScaler`.  
   - Ajuste de hiperparámetros mediante `GridSearchCV` con validación cruzada (k=3).  
   - Parámetros explorados: `C`, `kernel` (`linear`, `rbf`), `gamma`.  

2. **Random Forest**  
   - Modelo de ensamblado de árboles (`sklearn.ensemble.RandomForestClassifier`).  
   - Ajuste con rejilla de hiperparámetros (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).  
   - Permite extraer **importancia de características**, útil para interpretación.

3. **XGBoost**  
   - Clasificador de gradiente optimizado (`xgboost.XGBClassifier`).  
   - Parámetros: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.  
   - Entrenado con validación cruzada idéntica (cv=3, f1-weighted).  

El proceso de entrenamiento se centraliza en `src/models/train_models.py`, 
donde se definen los **grids**, la división de datos (80 % train / 20 % test), 
y se exportan los artefactos entrenados (`.joblib`) junto con los reportes JSON 
y las matrices de confusión.

---

## 3. Resultados obtenidos

Los valores de F1-macro confirman un desempeño moderado:

| Modelo | F1-macro | Accuracy |
|---------|-----------|-----------|
| SVM | **0.7399** | 0.76 |
| XGBoost | 0.6659 | 0.72 |
| Random Forest | 0.6229 | 0.69 |

La clase *caminar* es la más estable, mientras que *girar* presenta baja recall en dos de los tres modelos, 
posiblemente por pérdida de landmarks durante la rotación corporal.

---

*(Se deja la sección de conclusiones abierta para futuras ejecuciones del equipo.)*
